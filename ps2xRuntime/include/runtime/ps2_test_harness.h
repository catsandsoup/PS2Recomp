#pragma once

// Test and replay support, driven from guest vblanks so that runs are reproducible (use with
// RT_TIME=virtual):
//  - Pad input reaches the game only at vblank boundaries. Each vblank takes the next state from a
//    movie (RT_MOVIE_PLAY), a script (RT_INPUT_SCRIPT, times in seconds of guest time), the test
//    server, or the host's live input, in that order.
//  - RT_MOVIE_RECORD=<file> records the pad state the game saw, one line per change.
//  - RT_STATE_HASH=<file> logs hashes of guest memory every RT_STATE_HASH_INTERVAL vblanks
//    (default 60), to compare runs for determinism.
//  - RT_EXIT_AT_VBLANK=<n> stops the game after n vblanks.
//
// Movie format (text), version 2: "# roadtrip-movie 2" header, then "<vblank> <port> <buttons hex> <lx> <ly>
// <rx> <ry>" whenever a port's state changes, "# connect <vblank> <port> 0|1" when a pad is unplugged or
// plugged in, "# marker <vblank> <kind> <text>" lines, and "# end <vblank>". Version 1 movies (no port
// column) drive both ports with the same state. Buttons are active-low DualShock bits (0xFFFF = nothing
// pressed); sticks are 0..255 with 128 centred.

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class PS2Runtime;

namespace ps2_test
{
    struct PadState
    {
        uint16_t buttons = 0xFFFFu;
        uint8_t lx = 128, ly = 128, rx = 128, ry = 128;
        bool operator==(const PadState &) const = default;
    };

    constexpr int kPadPorts = 2;

    // Host input from the keyboard/gamepads (any thread). Used when nothing else drives the pads
    // (and never while a test client is attached). The one-argument form is port 0.
    void setLiveInput(const PadState &state);
    void setLiveInput(int port, const PadState &state);
    // Whether a pad is plugged into the port (port 0 defaults to yes, port 1 to no once the host
    // reports anything for it; until then both are connected, as before).
    void setLiveConnected(int port, bool connected);

    // Vibration: the motor speeds the game last set for a port (small: 0/1, large: 0..255),
    // for the host to poll (any thread). `changes` counts updates that changed the values.
    struct ActuatorState
    {
        uint8_t small = 0, large = 0;
        uint64_t changes = 0;
    };
    ActuatorState actuatorState(int port);
    // Called by the pad emulation (EE thread) whenever the game sends motor values.
    void onActuator(int port, uint8_t small, uint8_t large);

    // The vblank the game is at (for logs).
    uint64_t currentVblank();

    // Pauses the game (a host menu is open): the game thread waits at the next vblank until
    // resumed, so the picture holds and nothing advances. Ignored while a test client is attached.
    void setPaused(bool paused);
    bool paused();
    // Register before run(), clear after run() has joined its EE thread. Called
    // only on the EE thread inside the host-menu pause hold, with the harness
    // mutex released. The handler must not advance guest execution or throw.
    // A handler runs on entry and about every 100 ms while held; notify wakes it
    // sooner for app requests. Lockstep test-client parking does not call it.
    using PausedWorkHandler = std::function<void(PS2Runtime &)>;
    void setPausedWorkHandler(PausedWorkHandler handler);
    void notifyPausedWork();
    // While paused: let `vblanks` frames through (e.g. so a changed picture setting shows), then hold again.
    void stepPaused(uint32_t vblanks);
    // Adds a marker to the movie being recorded (any thread).
    void addMarker(const std::string &kind, const std::string &text = {});
    // True while a movie or script drives the pad.
    bool inputScripted();

    // Ends the movie being recorded ("# end <vblank>": after it, playback returns to live input;
    // without it the last state holds). Called on exit.
    void finishRecording();

    // Sound output (48 kHz stereo), for the test server's "audio" command.
    void onAudio(const int16_t *interleavedStereo, size_t frames);

    // Rendering on/off (VU1 microprograms and their drawing). Off only in test runs that don't
    // need pictures: the game's logic does not depend on it (verified with state hashes), and it
    // is the bulk of the cost in 3D scenes. RT_RENDER=0 starts with it off.
    bool renderingEnabled();
    void setRenderingEnabled(bool on);

    // Called on the EE thread at the start of every guest vblank.
    void onVblank(PS2Runtime &runtime, uint64_t vblank);
    // A save state was loaded (EE thread): the game is back at `vblank` (just after its events).
    // A lockstep client's run ends at the next vblank; a movie replays from there.
    void onStateLoaded(uint64_t vblank);

    // RT_TEST_SOCKET=<path>: a JSON-lines control server on a Unix socket for test drivers (one
    // client at a time). While a client is attached the game runs in lockstep: it parks at a vblank
    // until the client asks for more, so reads, writes, pad changes and frame grabs happen between
    // two exact vblanks. Commands (one JSON object per line, reply likewise):
    //   {"cmd":"run","vblanks":N}            run N vblanks, reply {"vblank":v} when parked again
    //   {"cmd":"pad","buttons":B,"lx":..}    pad state from the next vblank on (B active-low); with
    //                                        "port":0|1 only that port (else both), "connected":0|1
    //   {"cmd":"release_pad"}                back to movie/script/live input (both ports)
    //   {"cmd":"actuators"}                  motor values per port: small, large, changes
    //   {"cmd":"pad_info"}                   per port: open, analog, connected, align
    //   {"cmd":"read","space":"ee|spr|iop|vu1","addr":A,"len":L}  -> {"data":"<hex>"}
    //   {"cmd":"write","space":...,"addr":A,"data":"<hex>"}
    //   {"cmd":"frame","path":P}             the presented picture as raw RGBA -> {"width":W,"height":H}
    //   {"cmd":"stats"}                      vblank, presented frames, thread load
    //   {"cmd":"marker","kind":K,"text":T}   adds a marker to the movie being recorded
    //   {"cmd":"render","on":0|1}            rendering on/off (see renderingEnabled)
    //   {"cmd":"step","vblanks":N,"buttons":B,"lx":..,"port":P,"reads":"ee:ADDR:LEN,..."}
    //                                        pad + run + reads in one round trip -> {"vblank","data":[hex,..]}
    //   {"cmd":"audio"}                      sound since the last query: frames, rms, peak, hash,
    //                                        lr_diff (mean |left - right|)
    //   {"cmd":"quit"}
    //   anything else goes to the app's command handler (setCommandHandler), if any
    void startServerIfRequested(PS2Runtime &runtime);

    // The app's own test-socket commands: called (on the server thread, with the game parked
    // between runs) with the command name and the whole JSON line; returns the JSON reply, or an
    // empty string for a command it doesn't know. jsonField reads a top-level string or number.
    using CommandHandler = std::function<std::string(const std::string &cmd, const std::string &line)>;
    void setCommandHandler(CommandHandler handler);
    std::string jsonField(const std::string &line, const char *key);

    // Render thread: a frame grab is waiting (latch the picture even if the vblank is unchanged),
    // and hands the latched picture (tightly packed RGBA) to it.
    bool frameCaptureRequested();
    void deliverFrameCapture(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height);
    // The app's own picture of the game (a save state's thumbnail), any thread: ask, then poll.
    // The next latched picture is delivered (headless too, when the game renders), as the GS
    // scanned it out, before post-processing; empty if there was none.
    void requestAppFrameCapture();
    bool takeAppFrameCapture(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height);
}
