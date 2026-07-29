import React, { useEffect, useRef, useState } from "react";
import { mdToHtml } from "./format.js";

export default function ChatRail({ messages, sending, onSend }) {
  const [text, setText] = useState("");
  const threadRef = useRef(null);
  const taRef = useRef(null);

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  const submit = (e) => {
    e.preventDefault();
    const t = text.trim();
    if (!t || sending) return;
    onSend(t);
    setText("");
    if (taRef.current) taRef.current.style.height = "auto";
  };

  const onKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(e); }
  };
  const grow = (e) => {
    setText(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(120, e.target.scrollHeight) + "px";
  };

  return (
    <aside className="rail">
      <div className="rail-head">
        <div className="label">Direction</div>
        <div className="t">Tell the atelier what you want.</div>
      </div>

      <div className="thread" ref={threadRef}>
        {messages.map((m, i) => (
          <div className={"msg " + m.role} key={i}>
            <div className="who">{m.role === "user" ? "You" : "Atelier"}</div>
            <div className="bubble" dangerouslySetInnerHTML={{ __html: mdToHtml(m.text) }} />
            {m.steps?.length > 0 && (
              <div className="steps">
                {m.steps.map((s, j) => <div className="step" key={j}>· <b>{toolName(s)}</b> {tail(s)}</div>)}
              </div>
            )}
            {m.learned?.triples_added > 0 && (
              <div className="grew">↳ learned {m.learned.triples_added} facts · {m.learned.chunks_indexed} passages</div>
            )}
          </div>
        ))}
        {sending && (
          <div className="msg ai">
            <div className="who">Atelier</div>
            <div className="thinking">thinking<i /><i /><i /></div>
          </div>
        )}
      </div>

      <div className="composer">
        <form onSubmit={submit}>
          <textarea
            ref={taRef} rows={1} value={text} onChange={grow} onKeyDown={onKey}
            placeholder="Style me for a gallery opening…  ·  Is quiet luxury still trending?"
          />
          <button className="send" type="submit" disabled={sending || !text.trim()} aria-label="Send">
            <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"><path d="M7 11l5-5 5 5M12 6v13" /></svg>
          </button>
        </form>
        <div className="hint">Enter to send · Shift+Enter for a new line · drop a photo on the canvas to review a look</div>
      </div>
    </aside>
  );
}

function toolName(s = "") { const m = s.match(/^(\w+)\[/); return m ? m[1] : "step"; }
function tail(s = "") { const m = s.match(/\[(.+?)\]/); return m ? m[1].slice(0, 40) : ""; }
