import Foundation

/// Qwen3's chat template, rendered for a single user turn.
///
/// The xcframework is built without llama.cpp's `common` library, so the jinja
/// templating path is unavailable in-process and the rendered form is pinned here.
/// It matches what `jinja` produces from the template stored in the GGUF
/// (`tokenizer.chat_template`, byte-identical to the HF `tokenizer_config.json`)
/// for `add_generation_prompt=true`.
///
/// A prompt built this way must be tokenized with `parseSpecial: true`, or the
/// control tokens are encoded as literal text.
enum ChatTemplate {

    /// Thinking disabled: the assistant turn opens with an already-closed `<think>`
    /// block, so the model answers directly instead of spending its budget on a
    /// reasoning preamble.
    static func qwen3NoThink(_ content: String) -> String {
        "<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    /// Qwen3's default: no pre-closed block, so the model reasons first.
    static func qwen3Thinking(_ content: String) -> String {
        "<|im_start|>user\n\(content)<|im_end|>\n<|im_start|>assistant\n"
    }
}
