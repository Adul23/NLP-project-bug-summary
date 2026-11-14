import { useMemo } from "react";
import CodeMirror from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { Prec, RangeSetBuilder, type Extension } from "@codemirror/state";
import { Decoration, EditorView, keymap } from "@codemirror/view";
import { useDarkMode } from "../lib/useDarkMode";

const bugLine = Decoration.line({ class: "cm-bug-line" });

function highlightLine(line: number | null): Extension {
  if (line === null) return [];
  return EditorView.decorations.of((view) => {
    const builder = new RangeSetBuilder<Decoration>();
    if (line >= 1 && line <= view.state.doc.lines) {
      const from = view.state.doc.line(line).from;
      builder.add(from, from, bugLine);
    }
    return builder.finish();
  });
}

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit?: () => void;
  highlight?: number | null;
  minHeight?: string;
  maxHeight?: string;
  ariaLabel: string;
  placeholder?: string;
}

export function CodeEditor({
  value,
  onChange,
  onSubmit,
  highlight = null,
  minHeight = "280px",
  maxHeight = "520px",
  ariaLabel,
  placeholder,
}: Props) {
  const dark = useDarkMode();
  const extensions = useMemo(() => {
    const exts: Extension[] = [python(), highlightLine(highlight), EditorView.contentAttributes.of({ "aria-label": ariaLabel })];
    if (onSubmit) {
      exts.push(
        Prec.highest(
          keymap.of([
            {
              key: "Mod-Enter",
              run: () => {
                onSubmit();
                return true;
              },
            },
          ]),
        ),
      );
    }
    return exts;
  }, [highlight, onSubmit, ariaLabel]);

  return (
    <CodeMirror
      className="cm-theme-app"
      value={value}
      onChange={onChange}
      extensions={extensions}
      theme={dark ? "dark" : "light"}
      minHeight={minHeight}
      maxHeight={maxHeight}
      placeholder={placeholder}
      basicSetup={{ lineNumbers: true, foldGutter: false, highlightActiveLine: true, autocompletion: false }}
    />
  );
}
