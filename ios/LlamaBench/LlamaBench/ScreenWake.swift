import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Holds the screen awake while a long run is in progress.
///
/// A benchmark pass takes tens of minutes per variant. If the device locks, the
/// app stops being frontmost and its work is suspended, which stalls the run and
/// distorts the peak-RAM and throughput figures being measured. This is
/// independent of the device's Auto-Lock setting.
///
/// Callers scope it to an active run rather than leaving it on. iOS also clears
/// the flag when the app stops being frontmost.
enum ScreenWake {
    /// Sets the flag and returns the value UIKit holds afterwards, so a caller can
    /// record the effective state rather than the requested one.
    @discardableResult
    static func setDisabled(_ disabled: Bool) -> Bool {
        UIApplication.shared.isIdleTimerDisabled = disabled
        return UIApplication.shared.isIdleTimerDisabled
    }
}
