// A minimal iOS app that runs a Zyth program and shows what it printed.
// Built by scripts/ios-build.sh (swiftc, no Xcode project needed).
import SwiftUI
import Zyth

let program = """
fn fib(n: i64) -> i64 = if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
print("Hello from Zyth on iOS")
print("fib(20) =", fib(20))
print("squares:", (1..6).map(|x| x * x))
"""

/// Runs `src` with the embedded Zyth runtime: (succeeded, printed output, error text).
func runZyth(_ src: String) -> (Bool, String, String) {
    let bytes = Array(src.utf8)
    guard let buf = zyth_run(bytes, bytes.count) else { return (false, "", "no result") }
    defer { zyth_buf_free(buf) }
    var n: UInt64 = 0
    memcpy(&n, buf, 8)
    let data = Data(bytes: buf + 8, count: Int(n))
    let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
    return (obj["ok"] as? Bool ?? false, obj["output"] as? String ?? "", obj["error"] as? String ?? "")
}

@main
struct ZythDemoApp: App {
    let result = runZyth(program)
    init() {
        // also on the console (`xcrun simctl launch --console`)
        print(result.1, terminator: "")
        if !result.0 { print(result.2) }
    }
    var body: some Scene {
        WindowGroup {
            ScrollView {
                Text(result.1 + result.2)
                    .font(.system(.body, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
                    .accessibilityLabel("Program output")
            }
        }
    }
}
