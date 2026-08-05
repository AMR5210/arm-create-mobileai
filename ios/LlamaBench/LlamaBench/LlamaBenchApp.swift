import SwiftUI

@main
struct LlamaBenchApp: App {
    var body: some Scene {
        WindowGroup {
            TabView {
                ContentView()
                    .tabItem { Label("Benchmark", systemImage: "speedometer") }
                CompareView()
                    .tabItem { Label("Compare", systemImage: "text.bubble") }
            }
        }
    }
}
