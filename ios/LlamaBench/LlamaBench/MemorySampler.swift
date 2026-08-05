import Foundation

/// Samples the process's resident memory and retains the maximum observed.
///
/// Scope of the figure this produces: it is the whole app's resident size, which
/// includes the SwiftUI host and the linked framework as well as model weights
/// and KV cache. It is therefore comparable **across the three model variants**
/// measured in the same app — which is what the PTQ-vs-QAT-vs-fp16 comparison
/// needs — but not directly comparable to the desktop harness's figure, which is
/// `ru_maxrss` of a separate `llama-bench` child process. `scripts/benchmark.py`
/// records the desktop value; both are labelled in the emitted JSON.
///
/// Reset per variant, with the previous model fully released first, so one run's
/// allocations do not carry into the next.
final class MemorySampler: @unchecked Sendable {
    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "llamabench.memory-sampler")
    private let lock = NSLock()
    private var peak: UInt64 = 0

    /// Current resident size via `task_info` with the `MACH_TASK_BASIC_INFO` flavor.
    static func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(
            MemoryLayout<mach_task_basic_info>.size / MemoryLayout<natural_t>.size)
        let kerr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return kerr == KERN_SUCCESS ? info.resident_size : 0
    }

    /// Begins sampling at `interval` seconds, discarding any previous peak.
    func start(interval: TimeInterval = 0.1) {
        stop()
        lock.lock(); peak = 0; lock.unlock()
        sample()
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + interval, repeating: interval)
        t.setEventHandler { [weak self] in self?.sample() }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    /// Records one sample immediately. Called on teardown so a peak that occurs
    /// between timer ticks is still captured.
    func sample() {
        let r = MemorySampler.residentBytes()
        lock.lock()
        if r > peak { peak = r }
        lock.unlock()
    }

    var peakBytes: UInt64 {
        lock.lock(); defer { lock.unlock() }
        return peak
    }
}
