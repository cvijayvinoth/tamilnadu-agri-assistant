"""Browser-side read-aloud for assistant answers (Web Speech API).

Design notes
------------
* Synthesis happens in the **browser tab**, via ``window.speechSynthesis``.
  Nothing is sent to a server and no audio file is produced, so the privacy
  promise made in the sidebar ("no audio leaves this machine") stays true for
  output as well as input.
* This deliberately lives in ``src/ui/`` rather than ``src/voice.py``: that
  module documents itself as speech-to-text only, and cactus-needle/Whistle
  has no synthesis capability at all.
* Asymmetry worth knowing: speech INPUT is English-only (Whistle supports
  EN/DE/FR/ES/IT/NL/PL, not Tamil), but speech OUTPUT can read Tamil provided
  a ta-IN voice is installed in the operating system.
* Answers are markdown with inline ``[1]`` citation markers. Reading that raw
  sounds broken ("asterisk asterisk Seeds asterisk asterisk, bracket one"),
  so ``clean_for_speech`` is the substantive part of this module.
"""

from __future__ import annotations

import json
import re

from config import (
    TTS_LANG_PREFS,
    TTS_LOCAL_VOICES_ONLY,
    TTS_MAX_CHARS,
    TTS_PITCH,
    TTS_RATE,
)

# Fenced code blocks and inline code: read as prose, not syntax.
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
# Markdown links/images -> keep the visible label, drop the target.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Inline citation markers such as [1] or [2, 3] produced by the answer chain.
_CITATION_RE = re.compile(r"\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")
# Bare URLs would otherwise be spelled out character by character.
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.DOTALL)
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", re.MULTILINE)
_WS_RE = re.compile(r"[ \t]{2,}")
_BLANKS_RE = re.compile(r"\n{3,}")
# "farmers ." -> "farmers." after a citation/URL has been stripped out.
_ORPHAN_PUNCT_RE = re.compile(r"\s+([.,;:!?])")
# ".." / ". ." left behind when a marker sat between two sentences.
_DUP_PUNCT_RE = re.compile(r"([.,;:!?])(?:\s*\1)+")

# Covers the emoji actually used in this app's answers/UI plus the common
# pictographic ranges; speech engines either skip these or say "emoji".
_EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF"
    "\U0000FE00-\U0000FE0F"
    "\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF"
    "]+",
    flags=re.UNICODE,
)

_SENTENCE_END_RE = re.compile(r"[.!?](?:\s|$)")


def clean_for_speech(markdown: str, max_chars: int = TTS_MAX_CHARS) -> str:
    """Turn a markdown answer into something that sounds right when spoken.

    Strips syntax that would otherwise be read out literally, removes inline
    citation markers and URLs, and truncates on a sentence boundary so the
    reader never stops mid-word.
    """
    if not markdown:
        return ""

    text = str(markdown)
    text = _FENCE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _CITATION_RE.sub(" ", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _RULE_RE.sub(" ", text)
    text = _TABLE_SEP_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = text.replace("|", " ")
    text = _EMOJI_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n\n", text)
    # Removing citations/URLs leaves gaps like "farmers ." -> tidy them up,
    # otherwise the speech engine inserts an odd pause before the full stop.
    text = _ORPHAN_PUNCT_RE.sub(r"\1", text)
    text = _DUP_PUNCT_RE.sub(r"\1", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    if max_chars and len(text) > max_chars:
        head = text[:max_chars]
        # Prefer the last sentence end; fall back to the last space.
        ends = [m.end() for m in _SENTENCE_END_RE.finditer(head)]
        if ends and ends[-1] > max_chars * 0.5:
            text = head[: ends[-1]].strip()
        else:
            cut = head.rfind(" ")
            text = (head[:cut] if cut > 0 else head).strip() + "…"

    return text


_BTN = (
    "border:1px solid rgba(148,163,184,.45);background:#fff;color:#0b3d20;"
    "border-radius:999px;padding:.18rem .7rem;font:12px system-ui,sans-serif;"
    "cursor:pointer;"
)


def _js_payload(text: str, lang_prefs, rate: float, pitch: float,
                local_only: bool) -> str:
    """JSON-encode every value crossing into JS.

    Answer text is model output and may contain quotes, backslashes or
    newlines; ``json.dumps`` is what stops that breaking the script. ``</`` is
    split so a literal "</script>" inside an answer cannot end the block early.
    """
    blob = {
        "text": text,
        "langPrefs": list(lang_prefs),
        "rate": float(rate),
        "pitch": float(pitch),
        "localOnly": bool(local_only),
    }
    return json.dumps(blob, ensure_ascii=False).replace("</", "<\\/")


_SPEAK_TMPL = """
<div class="agri-speaker" data-nonce="__NONCE__">
  <div id="agri-tts-status" style="font:12px/1.5 system-ui,sans-serif;
       color:#64748b;padding:2px 0;"></div>
  <div style="display:flex;gap:.4rem;flex-wrap:wrap;">
    <button id="agri-tts-play"  style="__BTN__">&#9654; Play</button>
    <button id="agri-tts-pause" style="__BTN__">&#9208; Pause</button>
    <button id="agri-tts-stop"  style="__BTN__">&#9209; Stop</button>
  </div>
</div>
<script>
(function () {
  var CFG = __CFG__;
  var synth = window.speechSynthesis;
  var statusEl = document.getElementById("agri-tts-status");

  if (!synth || typeof SpeechSynthesisUtterance === "undefined") {
    statusEl.textContent = "This browser does not support speech synthesis.";
    return;
  }

  function pickVoice() {
    var voices = synth.getVoices() || [];
    if (CFG.localOnly) {
      var local = voices.filter(function (v) { return v.localService; });
      if (local.length) { voices = local; }
    }
    var i, j;
    for (i = 0; i < CFG.langPrefs.length; i++) {
      var want = CFG.langPrefs[i].toLowerCase();
      for (j = 0; j < voices.length; j++) {
        if ((voices[j].lang || "").toLowerCase().replace("_", "-") === want) {
          return voices[j];
        }
      }
    }
    // Fall back to the language family (any "en-*") before giving up.
    for (i = 0; i < CFG.langPrefs.length; i++) {
      var fam = CFG.langPrefs[i].split("-")[0].toLowerCase();
      for (j = 0; j < voices.length; j++) {
        if ((voices[j].lang || "").toLowerCase().indexOf(fam) === 0) {
          return voices[j];
        }
      }
    }
    return voices[0] || null;
  }

  function speak() {
    synth.cancel();                    // never queue on top of an old run
    var u = new SpeechSynthesisUtterance(CFG.text);
    var v = pickVoice();
    if (v) { u.voice = v; u.lang = v.lang; }
    u.rate = CFG.rate;
    u.pitch = CFG.pitch;
    u.onstart = function () {
      statusEl.textContent = "Reading aloud" + (v ? " \\u00b7 " + v.name : "");
    };
    u.onend = function () { statusEl.textContent = ""; };
    u.onerror = function (e) {
      statusEl.textContent = "Could not read aloud: " + (e.error || "unknown");
    };
    synth.speak(u);
  }

  document.getElementById("agri-tts-play").onclick = speak;
  document.getElementById("agri-tts-stop").onclick = function () {
    synth.cancel(); statusEl.textContent = "";
  };
  document.getElementById("agri-tts-pause").onclick = function () {
    if (synth.paused) { synth.resume(); }
    else if (synth.speaking) { synth.pause(); }
  };

  if (__AUTO__) {
    // Chrome loads the voice list asynchronously; wait for it once.
    if ((synth.getVoices() || []).length) {
      speak();
    } else {
      synth.onvoiceschanged = function () {
        synth.onvoiceschanged = null; speak();
      };
      setTimeout(speak, 600);          // Safari never fires onvoiceschanged
    }
  }
})();
</script>
"""


def speak_html(text: str, lang_prefs=None, rate: float = TTS_RATE,
               pitch: float = TTS_PITCH,
               local_only: bool = TTS_LOCAL_VOICES_ONLY,
               autoplay: bool = True, nonce: str = "") -> str:
    """Return an HTML/JS snippet that speaks ``text`` in the browser.

    Pass the result to ``streamlit.components.v1.html``. Voice selection walks
    ``lang_prefs`` in order and picks the first installed match, so a missing
    ta-IN voice degrades to an English voice rather than throwing.
    """
    cleaned = clean_for_speech(text)
    if not cleaned:
        return "<div></div>"

    payload = _js_payload(
        cleaned, lang_prefs or TTS_LANG_PREFS, rate, pitch, local_only
    )
    auto = "true" if autoplay else "false"
    return _SPEAK_TMPL.replace("__NONCE__", str(nonce)) \
                      .replace("__BTN__", _BTN) \
                      .replace("__CFG__", payload) \
                      .replace("__AUTO__", auto)


_PROBE_TMPL = """
<div style="font:12px/1.6 system-ui,sans-serif;color:#334155;">
  <div id="agri-voice-list">Detecting installed voices&hellip;</div>
</div>
<script>
(function () {
  var synth = window.speechSynthesis;
  var el = document.getElementById("agri-voice-list");
  if (!synth) { el.textContent = "Speech synthesis unsupported."; return; }
  function render() {
    var vs = synth.getVoices() || [];
    if (!vs.length) { el.textContent = "No voices detected yet."; return; }
    var ta = vs.filter(function (v) {
      return (v.lang || "").toLowerCase().indexOf("ta") === 0;
    });
    el.innerHTML =
      "<b>" + vs.length + "</b> voice(s) available. " +
      (ta.length
        ? "Tamil: <b>" + ta.map(function (v) { return v.name; }).join(", ") + "</b>"
        : "<span style='color:#b45309;'>No Tamil (ta-IN) voice installed &mdash; " +
          "Tamil text will be read by an English voice. Add one via " +
          "Settings &rarr; Time &amp; Language &rarr; Speech.</span>");
  }
  render();
  synth.onvoiceschanged = render;
})();
</script>
"""


def voice_probe_html() -> str:
    """Tiny panel listing the browser's installed voices.

    Worth having because "Tamil does not work" is almost always "no ta-IN voice
    is installed in the OS" rather than an application bug.
    """
    return _PROBE_TMPL


