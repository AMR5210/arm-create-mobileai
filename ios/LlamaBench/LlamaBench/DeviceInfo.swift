import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Supplies the `device` string for a result record.
///
/// `UIDevice.current.model` returns the device family ("iPhone") rather than the
/// specific model, while the record schema in `ios/README.md` calls for the real
/// model, since that string appears in the published results table. The hardware
/// identifier from `hw.machine` is specific, so it is used as the source of truth
/// and mapped to a marketing name.
///
/// The identifier is carried alongside the mapped name so a record stays
/// unambiguous. An identifier absent from the table resolves to the raw value,
/// and the table lists only models confirmed against hardware available to this
/// project.
enum DeviceInfo {

    /// Hardware identifier, e.g. `iPhone18,2`.
    static var hardwareIdentifier: String {
        #if targetEnvironment(simulator)
        // On a simulator `hw.machine` describes the host Mac, so the simulated
        // model comes from the environment the simulator provides.
        return ProcessInfo.processInfo.environment["SIMULATOR_MODEL_IDENTIFIER"]
            ?? "simulator"
        #else
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        guard size > 0 else { return "unknown" }
        var chars = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &chars, &size, nil, 0)
        return String(cString: chars)
        #endif
    }

    static let marketingNames: [String: String] = [
        "iPhone18,2": "iPhone 17 Pro Max",
        "iPhone13,2": "iPhone 12",
    ]

    static var marketingName: String {
        let id = hardwareIdentifier
        return marketingNames[id] ?? id
    }

    /// Value written to a record's `device` field.
    ///
    /// A simulator label states so explicitly: simulator figures run on
    /// desktop-class silicon and are not reportable device results, and the
    /// `device` string is what distinguishes them in the results table.
    static var recordLabel: String {
        #if targetEnvironment(simulator)
        return "\(marketingName) Simulator (desktop host) - DEV ONLY, not a reportable device"
        #else
        return "\(marketingName) (\(hardwareIdentifier))"
        #endif
    }
}
