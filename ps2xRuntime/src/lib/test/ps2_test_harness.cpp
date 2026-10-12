#include "runtime/ps2_test_harness.h"

#include "ps2_runtime.h"
#include "Stubs/Pad.h"
#include "runtime/ps2_save_state.h"
#include "ps2x/state_archive.h"

#include <algorithm>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <thread>
#ifdef _WIN32
// Winsock: the control socket is TCP on the loopback ("tcp:<port>"), since Python on Windows has no AF_UNIX.
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>
#include <unistd.h>
#endif
#include <cstdio>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string_view>
#include <vector>


namespace
{
#ifdef _WIN32
    using SockT = SOCKET;
    constexpr SockT kBadSock = INVALID_SOCKET;
    int sockRead(SockT s, void *buf, size_t n) { return ::recv(s, static_cast<char *>(buf), static_cast<int>(n), 0); }
    int sockWrite(SockT s, const void *buf, size_t n) { return ::send(s, static_cast<const char *>(buf), static_cast<int>(n), 0); }
    void sockClose(SockT s) { ::closesocket(s); }
    bool sockWaitReadable(SockT s, int timeoutMs)
    {
        WSAPOLLFD p{s, POLLRDNORM, 0};
        return ::WSAPoll(&p, 1, timeoutMs) > 0;
    }
#else
    using SockT = int;
    constexpr SockT kBadSock = -1;
    int sockRead(SockT s, void *buf, size_t n) { return static_cast<int>(::read(s, buf, n)); }
    int sockWrite(SockT s, const void *buf, size_t n) { return static_cast<int>(::write(s, buf, n)); }
    void sockClose(SockT s) { ::close(s); }
    bool sockWaitReadable(SockT s, int timeoutMs)
    {
        pollfd p{s, POLLIN, 0};
        return ::poll(&p, 1, timeoutMs) > 0;
    }
#endif
}

namespace ps2_stubs
{
    void setPadOverridePort(int port, uint16_t buttons, uint8_t lx, uint8_t ly, uint8_t rx, uint8_t ry,
                            bool connected);
}

namespace ps2_test
{
    namespace
    {
        struct ButtonName
        {
            const char *name;
            uint16_t mask;
        };
        constexpr ButtonName kButtons[] = {
            {"select", 0x0001}, {"l3", 0x0002}, {"r3", 0x0004}, {"start", 0x0008}, {"up", 0x0010},
            {"right", 0x0020}, {"down", 0x0040}, {"left", 0x0080}, {"l2", 0x0100}, {"r2", 0x0200},
            {"l1", 0x0400}, {"r1", 0x0800}, {"triangle", 0x1000}, {"circle", 0x2000}, {"cross", 0x4000},
            {"square", 0x8000},
        };
        constexpr double kVblanksPerSecond = 60000.0 / 1001.0;

        struct MovieEntry
        {
            uint64_t vblank;
            int port;          // -1: both ports (version 1 movies)
            PadState state;
            int connect = -1;  // 0/1: a "# connect" line (pad unplugged / plugged in) instead of a state
        };

        // What both pads read at one vblank.
        struct Pads
        {
            PadState state[kPadPorts];
            bool connected[kPadPorts] = {true, true};
        };

        struct ScriptPress
        {
            uint64_t from, until;
            uint16_t mask;
        };

        std::mutex g_mutex;
        PadState g_live[kPadPorts];
        bool g_liveConnected[kPadPorts] = {true, true};
        bool g_configured = false;
        std::vector<MovieEntry> g_movie;
        size_t g_movieIndex = 0;  // entries before this one have been applied to g_moviePads
        Pads g_moviePads;
        uint64_t g_movieEnd = 0;
        std::vector<ScriptPress> g_script;
        FILE *g_record = nullptr;
        Pads g_lastRecorded;
        bool g_recordedAny = false;
        uint64_t g_currentVblank = 0;
        FILE *g_hashLog = nullptr;

        // Test server (lockstep).
        std::condition_variable g_parkCv;   // game thread waits here while parked
        std::condition_variable g_serverCv; // server waits for the game to park / a capture
        bool g_serverEnabled = false;
        std::atomic<bool> g_rendering{[] { const char *e = std::getenv("RT_RENDER"); return !(e && *e == '0'); }()};
        bool g_attached = false;
        std::atomic<bool> g_quitRequested{false}; // an orderly quit: shut down normally
        uint64_t g_runTarget = 0;           // park when reaching this vblank
        uint64_t g_parkedAt = UINT64_MAX;   // vblank the game is parked at, or UINT64_MAX
        bool g_serverPadActive = false;
        // RT_TEST_LIVE=1: the control socket without lockstep (the game runs in real time and the
        // host menu pauses it), for driving the app's own UI on a device.
        bool g_liveSocket = false;
        Pads g_serverPads;
        bool g_paused = false; // a host menu is open
        PausedWorkHandler g_pausedWorkHandler;
        bool g_pausedWorkPending = false;
        uint32_t g_pausedSteps = 0; // frames to let through while paused
        // Vibration per port (any thread reads, the EE thread writes).
        std::mutex g_actMutex;
        ActuatorState g_actuators[kPadPorts];
        // Sound since the last "audio" query.
        uint64_t g_audioFrames = 0;
        double g_audioSquares = 0.0;
        int g_audioPeak = 0;
        uint64_t g_audioHash = 0xcbf29ce484222325ull;
        double g_audioLrDiff = 0.0; // sum of |left - right|: 0 for mono sound
        // Frame capture hand-off with the render thread.
        bool g_captureRequested = false;
        bool g_captureDone = false;
        std::vector<uint8_t> g_capture;
        // The app's own picture requests (requestAppFrameCapture), apart from the socket's.
        bool g_appCaptureRequested = false;
        bool g_appCaptureDone = false;
        std::vector<uint8_t> g_appCapture;
        uint32_t g_appCaptureWidth = 0, g_appCaptureHeight = 0;
        uint32_t g_captureWidth = 0, g_captureHeight = 0;
        uint64_t g_hashInterval = 60;

        uint64_t hashBytes(const uint8_t *data, size_t size, uint64_t h = 0xcbf29ce484222325ull)
        {
            // 64-bit FNV-1a over 8-byte words: fast enough for 32 MB once a second.
            size_t i = 0;
            for (; i + 8 <= size; i += 8)
            {
                uint64_t w;
                std::memcpy(&w, data + i, 8);
                h = (h ^ w) * 0x100000001b3ull;
            }
            for (; i < size; ++i)
                h = (h ^ data[i]) * 0x100000001b3ull;
            return h;
        }

        bool loadMovie(const char *path)
        {
            std::ifstream in(path);
            if (!in)
                return false;
            g_movieEnd = UINT64_MAX; // without an end line the last state holds forever
            int version = 1;
            for (std::string line; std::getline(in, line);)
            {
                if (line.rfind("# roadtrip-movie ", 0) == 0)
                {
                    version = std::atoi(line.c_str() + 17);
                    continue;
                }
                if (line.rfind("# end ", 0) == 0)
                {
                    g_movieEnd = std::strtoull(line.c_str() + 6, nullptr, 10);
                    continue;
                }
                if (line.rfind("# connect ", 0) == 0)
                {
                    MovieEntry e{};
                    unsigned long long vblank = 0;
                    int port = 0, connected = 1;
                    if (std::sscanf(line.c_str() + 10, "%llu %d %d", &vblank, &port, &connected) == 3 && port >= 0 &&
                        port < kPadPorts)
                    {
                        e.vblank = vblank;
                        e.port = port;
                        e.connect = connected ? 1 : 0;
                        g_movie.push_back(e);
                    }
                    continue;
                }
                if (line.empty() || line[0] == '#')
                    continue;
                std::istringstream fields(line);
                MovieEntry e{};
                e.port = -1;
                if (version >= 2)
                    fields >> e.vblank >> e.port;
                else
                    fields >> e.vblank;
                unsigned buttons = 0, lx = 128, ly = 128, rx = 128, ry = 128;
                fields >> std::hex >> buttons >> std::dec >> lx >> ly >> rx >> ry;
                if (!fields || e.port >= kPadPorts)
                    continue;
                e.state = PadState{static_cast<uint16_t>(buttons), static_cast<uint8_t>(lx), static_cast<uint8_t>(ly),
                                   static_cast<uint8_t>(rx), static_cast<uint8_t>(ry)};
                g_movie.push_back(e);
            }
            return true;
        }

        void loadScript(const char *spec)
        {
            // "<seconds>:<button>[:<hold seconds>],..." in guest time.
            std::istringstream items(spec);
            for (std::string item; std::getline(items, item, ',');)
            {
                std::istringstream parts(item);
                std::string at, name, hold;
                std::getline(parts, at, ':');
                std::getline(parts, name, ':');
                std::getline(parts, hold, ':');
                for (const auto &b : kButtons)
                {
                    if (name != b.name)
                        continue;
                    const double start = std::atof(at.c_str());
                    const double length = hold.empty() ? 0.25 : std::atof(hold.c_str());
                    const auto from = static_cast<uint64_t>(start * kVblanksPerSecond);
                    g_script.push_back({from, from + std::max<uint64_t>(1, static_cast<uint64_t>(length * kVblanksPerSecond)), b.mask});
                }
            }
            std::fprintf(stderr, "[input] script: %zu presses\n", g_script.size());
        }

        void configureLocked()
        {
            if (g_configured)
                return;
            g_configured = true;
            if (const char *play = std::getenv("RT_MOVIE_PLAY"))
            {
                if (loadMovie(play))
                    std::fprintf(stderr, "[movie] playing %s (%zu changes, ends at vblank %lld)\n", play, g_movie.size(),
                                 g_movieEnd == UINT64_MAX ? -1ll : static_cast<long long>(g_movieEnd));
                else
                    std::fprintf(stderr, "[movie] cannot read %s\n", play);
            }
            if (const char *script = std::getenv("RT_INPUT_SCRIPT"))
                loadScript(script);
            if (const char *record = std::getenv("RT_MOVIE_RECORD"))
            {
                g_record = std::fopen(record, "w");
                if (g_record)
                    std::fprintf(g_record, "# roadtrip-movie 2\n");
            }
            if (const char *hash = std::getenv("RT_STATE_HASH"))
            {
                g_hashLog = std::fopen(hash, "w");
                if (const char *interval = std::getenv("RT_STATE_HASH_INTERVAL"))
                    g_hashInterval = std::max<uint64_t>(1, std::strtoull(interval, nullptr, 10));
            }
        }

        Pads nextStateLocked(uint64_t vblank, bool &scripted)
        {
            scripted = false;
            if (!g_movie.empty() && vblank < g_movieEnd)
            {
                while (g_movieIndex < g_movie.size() && g_movie[g_movieIndex].vblank <= vblank)
                {
                    const MovieEntry &e = g_movie[g_movieIndex++];
                    for (int port = 0; port < kPadPorts; ++port)
                    {
                        if (e.port != -1 && e.port != port)
                            continue;
                        if (e.connect >= 0)
                            g_moviePads.connected[port] = e.connect != 0;
                        else
                            g_moviePads.state[port] = e.state;
                    }
                }
                scripted = true;
                return g_moviePads;
            }
            if (g_serverPadActive)
            {
                scripted = true;
                return g_serverPads;
            }
            Pads pads;
            for (int port = 0; port < kPadPorts; ++port)
            {
                pads.state[port] = g_live[port];
                pads.connected[port] = g_liveConnected[port];
            }
            if (!g_script.empty())
            {
                // A script presses buttons on both pads (as before per-port input); live input
                // still works alongside it (e.g. to take over afterwards).
                scripted = true;
                PadState s{};
                for (const auto &p : g_script)
                    if (vblank >= p.from && vblank < p.until)
                        s.buttons &= static_cast<uint16_t>(~p.mask);
                for (int port = 0; port < kPadPorts; ++port)
                {
                    const uint16_t live = g_live[port].buttons;
                    pads.state[port] = s;
                    pads.state[port].buttons &= live;
                }
            }
            return pads;
        }

        void logStateHash(PS2Runtime &runtime, uint64_t vblank)
        {
            // RT_STATE_HASH_FULL=1: one hash per save-state chunk (scheduler, VRAM, VU1, IOP,
            // SPU2, HLE stubs... see ps2_save_state.h).
            static const bool full = [] { const char *e = std::getenv("RT_STATE_HASH_FULL"); return e && *e == '1'; }();
            if (full)
            {
                std::string line = std::to_string(vblank);
                char buf[40];
                for (const auto &[name, hash] : ps2_save_state::chunkHashes(runtime))
                {
                    std::snprintf(buf, sizeof(buf), " %s=%016llx", name.c_str(), static_cast<unsigned long long>(hash));
                    line += buf;
                }
                std::fprintf(g_hashLog, "%s\n", line.c_str());
                std::fflush(g_hashLog);
                return;
            }
            PS2Memory &memory = runtime.memory();
            memory.syncGifVif1();
            const uint64_t ee = hashBytes(memory.getRDRAM(), PS2_RAM_SIZE);
            const uint64_t spr = hashBytes(memory.getScratchpad(), PS2_SCRATCHPAD_SIZE);
            const uint64_t vu1 = hashBytes(memory.getVU1Data(), PS2_VU1_DATA_SIZE);
            static std::vector<uint8_t> iop(2u * 1024u * 1024u);
            const uint64_t iopHash = runtime.readIopMemory(0, iop.data(), iop.size()) ? hashBytes(iop.data(), iop.size()) : 0u;
            std::fprintf(g_hashLog, "%llu ee=%016llx spr=%016llx vu1=%016llx iop=%016llx\n",
                         static_cast<unsigned long long>(vblank), static_cast<unsigned long long>(ee),
                         static_cast<unsigned long long>(spr), static_cast<unsigned long long>(vu1),
                         static_cast<unsigned long long>(iopHash));
            std::fflush(g_hashLog);
        }
    }

    void finishRecording()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_record)
            return;
        std::fprintf(g_record, "# end %llu\n", static_cast<unsigned long long>(g_currentVblank));
        std::fclose(g_record);
        g_record = nullptr;
    }

    void setLiveInput(const PadState &state) { setLiveInput(0, state); }

    void setLiveInput(int port, const PadState &state)
    {
        if (port < 0 || port >= kPadPorts)
            return;
        std::lock_guard<std::mutex> lock(g_mutex);
        g_live[port] = state;
    }

    void setLiveConnected(int port, bool connected)
    {
        if (port < 0 || port >= kPadPorts)
            return;
        std::lock_guard<std::mutex> lock(g_mutex);
        g_liveConnected[port] = connected;
    }

    ActuatorState actuatorState(int port)
    {
        if (port < 0 || port >= kPadPorts)
            return {};
        std::lock_guard<std::mutex> lock(g_actMutex);
        return g_actuators[port];
    }

    void onActuator(int port, uint8_t small, uint8_t large)
    {
        if (port < 0 || port >= kPadPorts)
            return;
        std::lock_guard<std::mutex> lock(g_actMutex);
        ActuatorState &a = g_actuators[port];
        if (a.small != small || a.large != large)
        {
            a.small = small;
            a.large = large;
            ++a.changes;
        }
    }

    void setPaused(bool paused)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_paused == paused)
            return;
        g_paused = paused;
        g_parkCv.notify_all();
    }

    void setPausedWorkHandler(PausedWorkHandler handler)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_pausedWorkHandler = std::move(handler);
        g_pausedWorkPending = true;
        g_parkCv.notify_all();
    }

    void notifyPausedWork()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_pausedWorkPending = true;
        g_parkCv.notify_all();
    }

    void stepPaused(uint32_t vblanks)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_pausedSteps += vblanks;
        g_parkCv.notify_all();
    }

    bool paused()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_paused;
    }

    uint64_t currentVblank()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_currentVblank;
    }

    void addMarker(const std::string &kind, const std::string &text)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_record)
        {
            std::fprintf(g_record, "# marker %llu %s %s\n", static_cast<unsigned long long>(g_currentVblank), kind.c_str(),
                         text.c_str());
            std::fflush(g_record);
        }
    }

    bool inputScripted()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return !g_movie.empty() || !g_script.empty();
    }

    void onVblank(PS2Runtime &runtime, uint64_t vblank)
    {
        Pads pads;
        {
            std::unique_lock<std::mutex> lock(g_mutex);
            configureLocked();
            g_currentVblank = vblank;
            // Lockstep: with a test server, park at the target vblank until the client asks for
            // more (or detaches). The game starts parked at vblank 1 so runs begin identically.
            if (g_serverEnabled && !g_liveSocket && (!g_attached || vblank >= g_runTarget))
            {
                g_parkedAt = vblank;
                g_serverCv.notify_all();
                g_parkCv.wait(lock, [&]
                              { return (g_attached && vblank < g_runTarget) || !g_serverEnabled || runtime.isStopRequested(); });
                g_parkedAt = UINT64_MAX;
            }
            // A host menu is open: hold the game here (the render thread keeps showing the last
            // picture and the menu) until it closes.
            while (g_paused && (!g_attached || g_liveSocket) && !runtime.isStopRequested())
            {
                if (g_pausedSteps > 0)
                {
                    --g_pausedSteps;
                    break;
                }
                // App work executes on this EE thread without advancing a guest
                // frame. Release the harness lock so the callback can query pause
                // state or accept another host request without lock inversion.
                // A notification arriving during work remains pending for the
                // next iteration instead of being lost before the wait.
                const auto handler = g_pausedWorkHandler;
                g_pausedWorkPending = false;
                if (handler)
                {
                    lock.unlock();
                    handler(runtime);
                    lock.lock();
                }
                // Wakes regularly for app views and window-close/stop handling.
                g_parkCv.wait_for(lock, std::chrono::milliseconds(100),
                                  [&] { return !g_paused || g_pausedSteps > 0 || g_pausedWorkPending ||
                                               (g_attached && !g_liveSocket) || runtime.isStopRequested(); });
            }
            g_currentVblank = vblank;
            bool scripted = false;
            pads = nextStateLocked(vblank, scripted);
            if (g_record)
            {
                const auto at = static_cast<unsigned long long>(vblank);
                for (int port = 0; port < kPadPorts; ++port)
                {
                    if (!g_recordedAny || pads.connected[port] != g_lastRecorded.connected[port])
                        std::fprintf(g_record, "# connect %llu %d %d\n", at, port, pads.connected[port] ? 1 : 0);
                    const PadState &st = pads.state[port];
                    if (!g_recordedAny || !(st == g_lastRecorded.state[port]))
                        std::fprintf(g_record, "%llu %d %04x %u %u %u %u\n", at, port, st.buttons, st.lx, st.ly, st.rx,
                                     st.ry);
                }
                std::fflush(g_record);
                g_lastRecorded = pads;
                g_recordedAny = true;
            }
        }
        for (int port = 0; port < kPadPorts; ++port)
        {
            const PadState &st = pads.state[port];
            ps2_stubs::setPadOverridePort(port, st.buttons, st.lx, st.ly, st.rx, st.ry, pads.connected[port]);
        }
        if (g_hashLog && vblank % g_hashInterval == 0u)
            logStateHash(runtime, vblank);
        // RT_EXIT_AT_VBLANK=<n>: stop the game after n guest vblanks (unattended runs).
        static const uint64_t exitAt = []
        { const char *e = std::getenv("RT_EXIT_AT_VBLANK"); return e ? std::strtoull(e, nullptr, 10) : 0ull; }();
        if (exitAt != 0u && vblank >= exitAt)
        {
            finishRecording();
            std::fprintf(stderr, "[test] reached vblank %llu, stopping\n", static_cast<unsigned long long>(vblank));
            runtime.requestStop();
        }
    }

    void onStateLoaded(uint64_t vblank)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_currentVblank = vblank;
        if (g_serverEnabled && !g_liveSocket && g_attached)
            g_runTarget = vblank + 1u;
        // A movie plays from the start again up to the vblank.
        g_movieIndex = 0;
        g_moviePads = Pads{};
    }

    namespace
    {
        CommandHandler g_commandHandler;

        std::string jsonValue(const std::string &line, const char *key)
        {
            const std::string needle = std::string("\"") + key + "\"";
            size_t at = line.find(needle);
            if (at == std::string::npos)
                return {};
            at = line.find(':', at + needle.size());
            if (at == std::string::npos)
                return {};
            ++at;
            while (at < line.size() && line[at] == ' ')
                ++at;
            if (at < line.size() && line[at] == '"')
            {
                const size_t end = line.find('"', at + 1);
                return line.substr(at + 1, end == std::string::npos ? std::string::npos : end - at - 1);
            }
            size_t end = at;
            while (end < line.size() && line[end] != ',' && line[end] != '}' && line[end] != ' ')
                ++end;
            return line.substr(at, end - at);
        }

        uint64_t jsonNumber(const std::string &line, const char *key, uint64_t fallback = 0)
        {
            const std::string v = jsonValue(line, key);
            return v.empty() ? fallback : std::strtoull(v.c_str(), nullptr, 0);
        }

        std::string toHex(const uint8_t *data, size_t size)
        {
            static const char *digits = "0123456789abcdef";
            std::string out(size * 2, '0');
            for (size_t i = 0; i < size; ++i)
            {
                out[2 * i] = digits[data[i] >> 4];
                out[2 * i + 1] = digits[data[i] & 15];
            }
            return out;
        }

        std::vector<uint8_t> fromHex(const std::string &hex)
        {
            std::vector<uint8_t> out(hex.size() / 2);
            for (size_t i = 0; i < out.size(); ++i)
                out[i] = static_cast<uint8_t>(std::strtoul(hex.substr(2 * i, 2).c_str(), nullptr, 16));
            return out;
        }

        // Guest memory for read/write: returns a host pointer for [addr, addr+len) or nullptr.
        uint8_t *memorySpace(PS2Runtime &runtime, const std::string &space, uint32_t addr, size_t len)
        {
            PS2Memory &memory = runtime.memory();
            if (space == "ee" || space.empty())
            {
                const uint32_t phys = addr & 0x1FFFFFFFu;
                return phys + len <= PS2_RAM_SIZE ? memory.getRDRAM() + phys : nullptr;
            }
            if (space == "spr")
            {
                const uint32_t off = addr & (PS2_SCRATCHPAD_SIZE - 1u);
                return off + len <= PS2_SCRATCHPAD_SIZE ? memory.getScratchpad() + off : nullptr;
            }
            if (space == "vu1")
            {
                memory.syncGifVif1();
                const uint32_t off = addr & (PS2_VU1_DATA_SIZE - 1u);
                return off + len <= PS2_VU1_DATA_SIZE ? memory.getVU1Data() + off : nullptr;
            }
            return nullptr;
        }

        // A "pad"/"step" command's pad fields: without "port" both pads get the state (as before
        // per-port input), with it only that one; "connected" plugs/unplugs it.
        void applyServerPadLocked(const std::string &line)
        {
            const PadState state{static_cast<uint16_t>(jsonNumber(line, "buttons", 0xFFFF)),
                                 static_cast<uint8_t>(jsonNumber(line, "lx", 128)),
                                 static_cast<uint8_t>(jsonNumber(line, "ly", 128)),
                                 static_cast<uint8_t>(jsonNumber(line, "rx", 128)),
                                 static_cast<uint8_t>(jsonNumber(line, "ry", 128))};
            const std::string portText = jsonValue(line, "port");
            const std::string connectedText = jsonValue(line, "connected");
            for (int port = 0; port < kPadPorts; ++port)
            {
                if (!portText.empty() && std::atoi(portText.c_str()) != port)
                    continue;
                if (!jsonValue(line, "buttons").empty() || jsonValue(line, "cmd") == "pad")
                    g_serverPads.state[port] = state;
                if (!connectedText.empty())
                    g_serverPads.connected[port] = std::atoi(connectedText.c_str()) != 0;
            }
            g_serverPadActive = true;
        }

        std::string handle(PS2Runtime &runtime, const std::string &line)
        {
            const std::string cmd = jsonValue(line, "cmd");
            if (cmd == "run" && g_liveSocket)
            {
                // Real time: just let that much time pass.
                const uint64_t count = jsonNumber(line, "vblanks", 1);
                std::this_thread::sleep_for(std::chrono::microseconds(count * 16667));
                return "{\"ok\":true,\"vblank\":" + std::to_string(g_currentVblank) + "}";
            }
            if (cmd == "run")
            {
                const uint64_t count = jsonNumber(line, "vblanks", 1);
                std::unique_lock<std::mutex> lock(g_mutex);
                if (g_parkedAt == UINT64_MAX)
                    g_serverCv.wait(lock, [&] { return g_parkedAt != UINT64_MAX || runtime.isStopRequested(); });
                const uint64_t from = g_parkedAt;
                g_runTarget = from + count;
                g_parkCv.notify_all();
                g_serverCv.wait(lock, [&]
                                { return (g_parkedAt != UINT64_MAX && g_parkedAt >= g_runTarget) || runtime.isStopRequested(); });
                if (runtime.isStopRequested())
                    return "{\"ok\":false,\"error\":\"stopped\"}";
                return "{\"ok\":true,\"vblank\":" + std::to_string(g_parkedAt) + "}";
            }
            if (cmd == "pad")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                applyServerPadLocked(line);
                return "{\"ok\":true}";
            }
            if (cmd == "release_pad")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_serverPadActive = false;
                g_serverPads = Pads{};
                return "{\"ok\":true}";
            }
            if (cmd == "actuators")
            {
                std::string reply = "{\"ok\":true,\"ports\":[";
                for (int port = 0; port < kPadPorts; ++port)
                {
                    const ActuatorState a = actuatorState(port);
                    reply += std::string(port ? "," : "") + "{\"small\":" + std::to_string(a.small) +
                             ",\"large\":" + std::to_string(a.large) + ",\"changes\":" + std::to_string(a.changes) + "}";
                }
                return reply + "]}";
            }
            if (cmd == "pad_info")
            {
                const ps2_stubs::PadDebugSnapshot snap = ps2_stubs::getPadDebugSnapshot();
                std::string reply = "{\"ok\":true,\"ports\":[";
                for (int port = 0; port < kPadPorts; ++port)
                {
                    const auto &p = snap.ports[port][0];
                    reply += std::string(port ? "," : "") + "{\"open\":" + (p.open ? "true" : "false") +
                             ",\"analog\":" + (p.analogMode ? "true" : "false") +
                             ",\"connected\":" + (p.connected ? "true" : "false") +
                             ",\"reads\":" + std::to_string(p.readCount) +
                             ",\"align\":\"" + toHex(p.actAlign, sizeof(p.actAlign)) + "\"}";
                }
                return reply + "]}";
            }
            if (cmd == "read" || cmd == "write")
            {
                const std::string space = jsonValue(line, "space");
                const auto addr = static_cast<uint32_t>(jsonNumber(line, "addr"));
                if (cmd == "read")
                {
                    const size_t len = jsonNumber(line, "len", 4);
                    std::vector<uint8_t> buffer(len);
                    bool ok = false;
                    if (space == "iop")
                        ok = runtime.readIopMemory(addr, buffer.data(), len);
                    else if (const uint8_t *src = memorySpace(runtime, space, addr, len))
                    {
                        std::memcpy(buffer.data(), src, len);
                        ok = true;
                    }
                    return ok ? "{\"ok\":true,\"data\":\"" + toHex(buffer.data(), len) + "\"}"
                              : "{\"ok\":false,\"error\":\"bad address\"}";
                }
                const std::vector<uint8_t> data = fromHex(jsonValue(line, "data"));
                bool ok = false;
                if (space == "iop")
                    ok = runtime.writeIopMemory(addr, data.data(), data.size());
                else if (uint8_t *dst = memorySpace(runtime, space, addr, data.size()))
                {
                    std::memcpy(dst, data.data(), data.size());
                    ok = true;
                }
                return ok ? "{\"ok\":true}" : "{\"ok\":false,\"error\":\"bad address\"}";
            }
            if (cmd == "frame")
            {
                const std::string path = jsonValue(line, "path");
                runtime.memory().syncGifVif1();
                std::unique_lock<std::mutex> lock(g_mutex);
                g_captureDone = false;
                g_captureRequested = true;
                if (!g_serverCv.wait_for(lock, std::chrono::seconds(10), [] { return g_captureDone; }))
                {
                    g_captureRequested = false;
                    return "{\"ok\":false,\"error\":\"no frame\"}";
                }
                if (path == "-")
                {
                    // Inline (a host harness driving a device can't read the device's files).
                    static const char *b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
                    std::string enc;
                    enc.reserve((g_capture.size() + 2) / 3 * 4);
                    for (size_t i = 0; i < g_capture.size(); i += 3)
                    {
                        uint32_t v = uint32_t(g_capture[i]) << 16;
                        if (i + 1 < g_capture.size())
                            v |= uint32_t(g_capture[i + 1]) << 8;
                        if (i + 2 < g_capture.size())
                            v |= g_capture[i + 2];
                        enc += b64[(v >> 18) & 63];
                        enc += b64[(v >> 12) & 63];
                        enc += i + 1 < g_capture.size() ? b64[(v >> 6) & 63] : '=';
                        enc += i + 2 < g_capture.size() ? b64[v & 63] : '=';
                    }
                    return "{\"ok\":true,\"width\":" + std::to_string(g_captureWidth) + ",\"height\":" +
                           std::to_string(g_captureHeight) + ",\"data\":\"" + enc + "\"}";
                }
                if (FILE *f = std::fopen(path.c_str(), "wb"))
                {
                    std::fwrite(g_capture.data(), 1, g_capture.size(), f);
                    std::fclose(f);
                }
                else
                    return "{\"ok\":false,\"error\":\"cannot write\"}";
                return "{\"ok\":true,\"width\":" + std::to_string(g_captureWidth) + ",\"height\":" +
                       std::to_string(g_captureHeight) + "}";
            }
            if (cmd == "stats")
            {
                PS2Memory &memory = runtime.memory();
                std::lock_guard<std::mutex> lock(g_mutex);
                return "{\"ok\":true,\"vblank\":" + std::to_string(g_currentVblank) +
                       ",\"flips\":" + std::to_string(memory.displayFlips()) +
                       ",\"vif1_busy_ns\":" + std::to_string(memory.gifVif1BusyNanos()) +
                       ",\"gs_busy_ns\":" + std::to_string(memory.gsThreadBusyNanos()) + "}";
            }
            if (cmd == "render")
            {
                setRenderingEnabled(jsonNumber(line, "on", 1) != 0);
                return "{\"ok\":true}";
            }
            if (cmd == "step")
            {
                // Pad (optional) + run + reads, one round trip for driving bots.
                if (!jsonValue(line, "buttons").empty() || !jsonValue(line, "connected").empty())
                {
                    std::lock_guard<std::mutex> lock(g_mutex);
                    applyServerPadLocked(line);
                }
                const std::string ran = handle(runtime, "{\"cmd\":\"run\",\"vblanks\":" +
                                                            std::to_string(jsonNumber(line, "vblanks", 1)) + "}");
                if (ran.find("\"ok\":true") == std::string::npos)
                    return ran;
                std::string reply = "{\"ok\":true,\"vblank\":" + std::to_string(g_parkedAt) + ",\"data\":[";
                std::istringstream reads(jsonValue(line, "reads"));
                bool first = true;
                for (std::string item; std::getline(reads, item, ',');)
                {
                    std::istringstream parts(item);
                    std::string space, addrText, lenText;
                    std::getline(parts, space, ':');
                    std::getline(parts, addrText, ':');
                    std::getline(parts, lenText, ':');
                    const auto addr = static_cast<uint32_t>(std::strtoull(addrText.c_str(), nullptr, 0));
                    const size_t len = std::strtoull(lenText.c_str(), nullptr, 0);
                    std::vector<uint8_t> buffer(len);
                    if (space == "iop")
                        runtime.readIopMemory(addr, buffer.data(), len);
                    else if (const uint8_t *src = memorySpace(runtime, space, addr, len))
                        std::memcpy(buffer.data(), src, len);
                    reply += std::string(first ? "" : ",") + "\"" + toHex(buffer.data(), len) + "\"";
                    first = false;
                }
                return reply + "]}";
            }
            if (cmd == "save_state" || cmd == "load_state")
            {
                // Save into / load from the in-memory slot, or the file "path", at the next savable
                // scheduler loop top, running vblank by vblank (at most "max_vblanks", default 120)
                // until it happened. A file save then waits for the file to be written ("raw":
                // true stores it uncompressed); a file load that is refused (damaged, another game,
                // a newer version) runs nothing. Load options: "resave_path" saves the machine
                // again right after loading; "test_fail": true makes the load fail its last check
                // (the machine is put back).
                const auto esc = [](const std::string &in)
                {
                    std::string out;
                    for (char c : in)
                    {
                        if (c == '"' || c == '\\')
                            out += '\\';
                        out += (static_cast<unsigned char>(c) < 0x20) ? ' ' : c;
                    }
                    return out;
                };
                const std::string path = jsonValue(line, "path");
                const uint64_t before = ps2_save_state::lastResult().sequence;
                const uint64_t writesBefore = ps2_save_state::lastWrite().sequence;
                if (cmd == "save_state")
                {
                    if (path.empty())
                        ps2_save_state::request(ps2_save_state::Op::Save);
                    else
                        ps2_save_state::requestSaveFile(path, jsonValue(line, "raw") != "true");
                }
                else if (path.empty())
                    ps2_save_state::request(ps2_save_state::Op::Load);
                else
                {
                    std::string error;
                    if (!ps2_save_state::requestLoadFile(path, error, jsonValue(line, "resave_path")))
                        return "{\"ok\":false,\"refused\":true,\"vblank\":" + std::to_string(g_parkedAt) + ",\"error\":\"" +
                               esc(error) + "\"}";
                    if (jsonValue(line, "test_fail") == "true") // (the game is parked: set before it runs)
                        ps2_save_state::failNextLoadForTest();
                }
                const uint64_t limit = jsonNumber(line, "max_vblanks", 120);
                std::string ran;
                for (uint64_t i = 0; i <= limit; ++i)
                {
                    ran = handle(runtime, "{\"cmd\":\"run\",\"vblanks\":1}");
                    if (ran.find("\"ok\":true") == std::string::npos || ps2_save_state::lastResult().sequence != before)
                        break;
                }
                const ps2_save_state::Result r = ps2_save_state::lastResult();
                if (r.sequence == before)
                {
                    ps2_save_state::cancel();
                    return "{\"ok\":false,\"error\":\"no savable point\",\"blockers\":\"" + r.blockers + "\"}";
                }
                std::string write;
                if (r.ok && ((cmd == "save_state" && !path.empty()) || !jsonValue(line, "resave_path").empty()))
                {
                    ps2_save_state::waitForWrites();
                    const ps2_save_state::WriteResult w = ps2_save_state::lastWrite();
                    char wbuf[256];
                    std::snprintf(wbuf, sizeof(wbuf), ",\"written\":%s,\"file_bytes\":%llu,\"write_ms\":%.2f,\"write_error\":\"%s\"",
                                  (w.sequence != writesBefore && w.ok) ? "true" : "false",
                                  static_cast<unsigned long long>(w.fileBytes), w.ms, esc(w.error).c_str());
                    write = wbuf;
                }
                char buf[768];
                std::snprintf(buf, sizeof(buf),
                              "{\"ok\":%s,\"vblank\":%llu,\"state_vblank\":%llu,\"loop_tops\":%llu,\"bytes\":%zu,"
                              "\"ms\":%.2f,\"order_mismatches\":%u,\"reserialized_equal\":%s,\"rolled_back\":%s,\"skipped\":\"%s\",\"blockers\":\"%s\",\"error\":\"%s\"",
                              r.ok ? "true" : "false", static_cast<unsigned long long>(g_parkedAt),
                              static_cast<unsigned long long>(r.vblank), static_cast<unsigned long long>(r.loopTops), r.bytes, r.ms,
                              r.orderMismatches, r.reserializedEqual ? "true" : "false", r.rolledBack ? "true" : "false",
                              r.skipped.c_str(), r.blockers.c_str(), esc(r.error).c_str());
                return std::string(buf) + write + "}";
            }
            if (cmd == "state_info")
            {
                // A state file's header: game, vblank, metadata and chunks.
                ps2_save_state::FileInfo info;
                std::string error;
                if (!ps2_save_state::readStateFileInfo(jsonValue(line, "path"), info, error))
                    return "{\"ok\":false,\"error\":\"" + error + "\"}";
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                              "{\"ok\":true,\"format\":%u,\"game_crc\":\"%08X\",\"vblank\":%llu,\"saved\":%lld,\"raw_bytes\":%llu,"
                              "\"file_bytes\":%llu,\"meta\":{",
                              info.formatVersion, info.gameCrc, static_cast<unsigned long long>(info.vblank),
                              static_cast<long long>(info.savedUnixTime), static_cast<unsigned long long>(info.rawSize),
                              static_cast<unsigned long long>(info.fileSize));
                std::string reply = buf;
                bool first = true;
                for (const auto &[k, v] : info.metadata)
                {
                    // Bytes (a thumbnail PNG) as "hex:<digits>"; quotes and backslashes escaped.
                    const bool binary = std::any_of(v.begin(), v.end(), [](char c)
                                                    { return static_cast<unsigned char>(c) < 0x20 || static_cast<unsigned char>(c) >= 0x7F; });
                    std::string text;
                    if (binary)
                        text = "hex:" + toHex(reinterpret_cast<const uint8_t *>(v.data()), v.size());
                    else
                        for (char c : v)
                        {
                            if (c == '"' || c == '\\')
                                text += '\\';
                            text += c;
                        }
                    reply += std::string(first ? "" : ",") + "\"" + k + "\":\"" + text + "\"";
                    first = false;
                }
                reply += "},\"chunks\":[";
                first = true;
                for (const auto &c : info.chunks)
                {
                    char cb[192];
                    std::snprintf(cb, sizeof(cb), "%s{\"name\":\"%s\",\"version\":%u,\"size\":%llu,\"hash\":\"%016llx\",\"digest\":\"%016llx\"}",
                                  first ? "" : ",", ps2x::fourccName(c.id).c_str(), c.version, static_cast<unsigned long long>(c.size),
                                  static_cast<unsigned long long>(c.hash), static_cast<unsigned long long>(c.digest));
                    reply += cb;
                    first = false;
                }
                return reply + "]}";
            }
            if (cmd == "state_hash")
            {
                // Per-chunk hashes of the state now (the game parked at a vblank).
                std::string reply = "{\"ok\":true,\"vblank\":" + std::to_string(g_parkedAt) + ",\"hashes\":{";
                bool first = true;
                for (const auto &[name, hash] : ps2_save_state::chunkHashes(runtime))
                {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), "%s\"%s\":\"%016llx\"", first ? "" : ",", name.c_str(),
                                  static_cast<unsigned long long>(hash));
                    reply += buf;
                    first = false;
                }
                return reply + "}}";
            }
            if (cmd == "state_dump")
            {
                // One chunk's bytes (as hashed by state_hash) to a file.
                const std::vector<uint8_t> bytes = ps2_save_state::chunkBytes(runtime, jsonValue(line, "chunk"));
                FILE *f = std::fopen(jsonValue(line, "path").c_str(), "wb");
                if (!f)
                    return "{\"ok\":false,\"error\":\"cannot write\"}";
                std::fwrite(bytes.data(), 1, bytes.size(), f);
                std::fclose(f);
                return "{\"ok\":true,\"size\":" + std::to_string(bytes.size()) + "}";
            }
            if (cmd == "state_chunks")
            {
                // The slot's chunks: name, size, hash.
                std::string reply = "{\"ok\":true,\"chunks\":[";
                bool first = true;
                for (const auto &[name, info] : ps2_save_state::slotChunks())
                {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "%s{\"name\":\"%s\",\"size\":%zu,\"hash\":\"%016llx\"}", first ? "" : ",",
                                  name.c_str(), info.first, static_cast<unsigned long long>(info.second));
                    reply += buf;
                    first = false;
                }
                return reply + "]}";
            }
            if (cmd == "marker")
            {
                addMarker(jsonValue(line, "kind"), jsonValue(line, "text"));
                return "{\"ok\":true}";
            }
            if (cmd == "audio")
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                const double rms = g_audioFrames ? std::sqrt(g_audioSquares / (2.0 * static_cast<double>(g_audioFrames))) : 0.0;
                char reply[256];
                std::snprintf(reply, sizeof(reply),
                              "{\"ok\":true,\"frames\":%llu,\"rms\":%.2f,\"peak\":%d,\"hash\":\"%016llx\",\"lr_diff\":%.3f}",
                              static_cast<unsigned long long>(g_audioFrames), rms, g_audioPeak,
                              static_cast<unsigned long long>(g_audioHash),
                              g_audioFrames ? g_audioLrDiff / static_cast<double>(g_audioFrames) : 0.0);
                g_audioFrames = 0;
                g_audioSquares = 0.0;
                g_audioPeak = 0;
                g_audioHash = 0xcbf29ce484222325ull;
                g_audioLrDiff = 0.0;
                return reply;
            }
            if (cmd == "pause")
            {
                // {"cmd":"pause","on":1|0}: holds the game as the app's in-game menu does
                // (setPaused). It parks at the next vblank only without lockstep (RT_TEST_LIVE=1).
                setPaused(jsonNumber(line, "on", 1) != 0);
                return "{\"ok\":true,\"vblank\":" + std::to_string(currentVblank()) + "}";
            }
            if (cmd == "quit")
            {
                g_quitRequested.store(true);
                runtime.requestStop();
                std::lock_guard<std::mutex> lock(g_mutex);
                g_parkCv.notify_all();
                return "{\"ok\":true}";
            }
            if (g_commandHandler)
            {
                std::string reply = g_commandHandler(cmd, line);
                if (!reply.empty())
                    return reply;
            }
            return "{\"ok\":false,\"error\":\"unknown command\"}";
        }

        void serveClient(PS2Runtime &runtime, SockT fd)
        {
            {
                std::lock_guard<std::mutex> lock(g_mutex);
                g_attached = true;
            }
            std::string buffer;
            char chunk[65536];
            for (;;)
            {
                const int n = sockRead(fd, chunk, sizeof(chunk));
                if (n <= 0)
                    break;
                buffer.append(chunk, static_cast<size_t>(n));
                size_t nl;
                while ((nl = buffer.find('\n')) != std::string::npos)
                {
                    const std::string line = buffer.substr(0, nl);
                    buffer.erase(0, nl + 1);
                    const std::string reply = handle(runtime, line) + "\n";
                    size_t sent = 0;
                    while (sent < reply.size())
                    {
                        const int w = sockWrite(fd, reply.data() + sent, reply.size() - sent);
                        if (w <= 0)
                            break;
                        sent += static_cast<size_t>(w);
                    }
                }
            }
            sockClose(fd);
            if (g_quitRequested.load())
                return; // the normal shutdown is under way (finishes a recorded movie, etc.)
            // The test driver is gone (it crashed or was killed): a test instance has no
            // reason to keep running, so it exits rather than linger in the background.
            std::fprintf(stderr, "[test] client disconnected; exiting\n");
            std::fflush(nullptr);
            std::_Exit(0);
        }

        // Test instances also exit if their parent process dies or no client ever attaches, so
        // a crashed or killed test run never leaves games behind.
        void startOrphanWatch()
        {
#ifdef _WIN32
            // No getppid on Windows: the harness ends its game with TerminateProcess, and the
            // client-gone path in serveClient exits a test instance whose driver is lost.
#else
            const pid_t parent = ::getppid();
            std::thread([parent]
                        {
                            for (;;)
                            {
                                std::this_thread::sleep_for(std::chrono::seconds(1));
                                if (::getppid() != parent)
                                {
                                    std::fprintf(stderr, "[test] parent process gone; exiting\n");
                                    std::fflush(nullptr);
                                    std::_Exit(0);
                                }
                            } })
                .detach();
#endif
        }
    }

    void startServerIfRequested(PS2Runtime &runtime)
    {
        const char *path = std::getenv("RT_TEST_SOCKET");
        if (!path)
            return;
#ifdef _WIN32
        WSADATA wsa;
        ::WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
        SockT listener = kBadSock;
        if (std::strncmp(path, "tcp:", 4) == 0)
        {
            // "tcp:<port>": the loopback only. Port 0 picks one and prints it.
            listener = ::socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in in{};
            in.sin_family = AF_INET;
            in.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            in.sin_port = htons(static_cast<uint16_t>(std::atoi(path + 4)));
            int yes = 1;
            ::setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&yes), sizeof(yes));
            if (listener == kBadSock || ::bind(listener, reinterpret_cast<sockaddr *>(&in), sizeof(in)) != 0 ||
                ::listen(listener, 1) != 0)
            {
                std::fprintf(stderr, "[test] cannot listen on %s\n", path);
                return;
            }
            socklen_t len = sizeof(in);
            ::getsockname(listener, reinterpret_cast<sockaddr *>(&in), &len);
            std::fprintf(stderr, "[test] control port %u\n", unsigned(ntohs(in.sin_port)));
        }
        else
        {
#ifdef _WIN32
            std::fprintf(stderr, "[test] RT_TEST_SOCKET on Windows must be tcp:<port>\n");
            return;
#else
            listener = ::socket(AF_UNIX, SOCK_STREAM, 0);
            sockaddr_un addr{};
            addr.sun_family = AF_UNIX;
            socklen_t addrLen = sizeof(addr);
            if (path[0] == '@')
            {
                // Abstract namespace (Linux, Android): no file; `adb forward tcp:N localabstract:<name>`
                // reaches it from a host, so a harness can drive the game on a device.
                const size_t n = std::min(std::strlen(path + 1), sizeof(addr.sun_path) - 1);
                std::memcpy(addr.sun_path + 1, path + 1, n);
                addrLen = socklen_t(offsetof(sockaddr_un, sun_path) + 1 + n);
            }
            else
            {
                std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", path);
                ::unlink(path);
            }
            if (listener < 0 || ::bind(listener, reinterpret_cast<sockaddr *>(&addr), addrLen) != 0 ||
                ::listen(listener, 1) != 0)
            {
                std::fprintf(stderr, "[test] cannot listen on %s\n", path);
                return;
            }
#endif
        }
        {
            std::lock_guard<std::mutex> lock(g_mutex);
            g_serverEnabled = true;
            const char *live = std::getenv("RT_TEST_LIVE");
            g_liveSocket = live && *live == '1';
        }
        std::fprintf(stderr, "[test] control socket %s%s\n", path, g_liveSocket ? " (live)" : "");
        startOrphanWatch();
        std::thread([&runtime, listener]
                    {
                        // One client for the whole run; the game waits at vblank 1 until it attaches.
                        if (!sockWaitReadable(listener, 300 * 1000))
                        {
                            std::fprintf(stderr, "[test] no client attached within 300 s; exiting\n");
                            std::fflush(nullptr);
                            std::_Exit(0);
                        }
                        const SockT fd = ::accept(listener, nullptr, nullptr);
                        sockClose(listener);
                        if (fd != kBadSock)
                            serveClient(runtime, fd); })
            .detach();
    }

    bool renderingEnabled() { return g_rendering.load(std::memory_order_relaxed); }
    void setRenderingEnabled(bool on) { g_rendering.store(on, std::memory_order_relaxed); }

    void onAudio(const int16_t *interleavedStereo, size_t frames)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_serverEnabled && !g_attached)
            return;
        for (size_t i = 0; i < frames; ++i)
            g_audioLrDiff += std::abs(static_cast<int>(interleavedStereo[2 * i]) - interleavedStereo[2 * i + 1]);
        for (size_t i = 0; i < frames * 2; ++i)
        {
            const int v = interleavedStereo[i];
            g_audioSquares += static_cast<double>(v) * v;
            g_audioPeak = std::max(g_audioPeak, v < 0 ? -v : v);
            g_audioHash = (g_audioHash ^ static_cast<uint16_t>(v)) * 0x100000001b3ull;
        }
        g_audioFrames += frames;
    }

    bool frameCaptureRequested()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        return g_captureRequested || g_appCaptureRequested;
    }

    void requestAppFrameCapture()
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_appCaptureRequested = true;
        g_appCaptureDone = false;
    }

    bool takeAppFrameCapture(std::vector<uint8_t> &rgba, uint32_t &width, uint32_t &height)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (!g_appCaptureDone)
            return false;
        g_appCaptureDone = false;
        rgba = std::move(g_appCapture);
        g_appCapture.clear();
        width = g_appCaptureWidth;
        height = g_appCaptureHeight;
        return true;
    }

    void deliverFrameCapture(const std::vector<uint8_t> &rgba, uint32_t width, uint32_t height)
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        if (g_appCaptureRequested)
        {
            g_appCapture = rgba;
            g_appCaptureWidth = width;
            g_appCaptureHeight = height;
            g_appCaptureRequested = false;
            g_appCaptureDone = true;
        }
        if (!g_captureRequested)
            return;
        g_capture = rgba;
        g_captureWidth = width;
        g_captureHeight = height;
        g_captureRequested = false;
        g_captureDone = true;
        g_serverCv.notify_all();
    }

    void setCommandHandler(CommandHandler handler) { g_commandHandler = std::move(handler); }

    std::string jsonField(const std::string &line, const char *key) { return jsonValue(line, key); }
}
