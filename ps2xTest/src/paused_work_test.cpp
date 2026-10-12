#include "ps2_runtime.h"
#include "runtime/ps2_test_harness.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

int main()
{
    PS2Runtime runtime;
    std::mutex mutex;
    std::condition_variable changed;
    std::atomic<unsigned> calls{0};
    std::atomic<bool> returned{false};
    std::thread::id executor;
    bool correctThread = true, held = true;
    ps2_test::setPausedWorkHandler([&](PS2Runtime &received) {
        // These take the harness mutex: servicing with it held would deadlock.
        const bool paused = ps2_test::paused();
        const auto frame = ps2_test::currentVblank();
        std::lock_guard lock(mutex);
        correctThread &= std::this_thread::get_id() == executor && &received == &runtime;
        held &= paused && frame == 37 && !returned.load();
        ++calls;
        changed.notify_all();
    });
    ps2_test::setPaused(true);
    std::thread ee([&] {
        executor = std::this_thread::get_id();
        ps2_test::onVblank(runtime, 37);
        returned = true;
    });
    bool serviced;
    {
        std::unique_lock lock(mutex);
        serviced = changed.wait_for(lock, std::chrono::seconds(2), [&] { return calls.load() > 0; });
    }
    const auto firstCalls = calls.load();
    ps2_test::notifyPausedWork();
    bool notified;
    {
        std::unique_lock lock(mutex);
        notified = changed.wait_for(lock, std::chrono::seconds(2), [&] { return calls.load() > firstCalls; });
    }
    const bool stayedHeld = !returned.load() && ps2_test::currentVblank() == 37;
    ps2_test::setPaused(false);
    ee.join();
    ps2_test::setPausedWorkHandler({});
    if (!serviced || !notified || !correctThread || !held || !stayedHeld || !returned.load())
    {
        std::cerr << "Paused host work was not serviced on the held EE boundary\n";
        return 1;
    }
    // No callback when no host pause is active; the ordinary vblank still returns.
    const auto count = calls.load();
    ps2_test::onVblank(runtime, 38);
    if (calls.load() != count) return 1;
    std::cout << "Paused host work checks passed\n";
}
