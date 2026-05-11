from pathlib import Path

TEMPLATE = """
{{- if or .System .Tools }}<|im_start|>system
{{ if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end -}}
<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{ if .Content }}{{ .Content }}
{{- else if .ToolCalls }}<tool_call>
{{ range .ToolCalls }}{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
{{ end }}</tool_call>
{{- end }}{{ if not $last }}<|im_end|>
{{ end }}
{{- else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}"""


def make_modelfile(mode: str) -> str:
    gguf_path = f"A:/contextual-research-agent/training/gguf_v2/{mode}/{mode}_v2-q4_k_m.gguf"

    return f"""FROM {gguf_path}
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.1
PARAMETER top_k 20
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096
"""


def main():
    base = Path("A:/contextual-research-agent/training/gguf_v2")
    modes = ["factual_qa"]

    for mode in modes:
        mode_dir = base / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        modelfile_path = mode_dir / "Modelfile"

        content = make_modelfile(mode)
        modelfile_path.write_text(content, encoding="utf-8")

        print(f"Created {modelfile_path}")


if __name__ == "__main__":
    main()
