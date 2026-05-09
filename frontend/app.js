console.log("NEW JS LOADED 🚀");
const form = document.getElementById("generator-form");
const contextInput = document.getElementById("context-input");
const countInput = document.getElementById("count-input");
const seedInput = document.getElementById("seed-input");
const minLenInput = document.getElementById("minlen-input");
const maxLenInput = document.getElementById("maxlen-input");
const reqSymbolInput = document.getElementById("req-symbol-input");
const reqDigitInput = document.getElementById("req-digit-input");
const statusText = document.getElementById("status-text");
const resultsCard = document.getElementById("results-card");
const resultsList = document.getElementById("results-list");
const copyAllButton = document.getElementById("copy-all-button");
const downloadButton = document.getElementById("download-button");
const auditPasswordInput = document.getElementById("audit-password");
const auditButton = document.getElementById("audit-button");
const auditStatus = document.getElementById("audit-status");

const pdfForm = document.getElementById("pdf-form");
const pdfPassword = document.getElementById("pdf-password");
const pdfFilename = document.getElementById("pdf-filename");
const pdfContent = document.getElementById("pdf-content");
const pdfStatus = document.getElementById("pdf-status");

function setStatus(message, isError = false) {
  statusText.textContent = message || "";
  statusText.classList.toggle("status-text--error", Boolean(isError));
}

function setAuditStatus(message, kind = "info") {
  if (!auditStatus) return;
  auditStatus.textContent = message || "";
  auditStatus.classList.toggle("status-text--error", kind === "error");
  auditStatus.classList.toggle("status-text--success", kind === "success");
  auditStatus.classList.toggle("status-text--info", kind === "info");
}

function setPdfStatus(message, isError = false) {
  if (!pdfStatus) return;
  pdfStatus.textContent = message || "";
  pdfStatus.classList.toggle("status-text--error", Boolean(isError));
}

function setLoading(isLoading) {
  const submitButton = form.querySelector("button[type='submit']");
  if (!submitButton) return;
  submitButton.disabled = isLoading;
  submitButton.textContent = isLoading ? "Generating..." : "Generate passwords";
}

function renderResults(passwords) {
  resultsList.innerHTML = "";
  // setAuditStatus("", "info");
  if (!passwords || passwords.length === 0) {
    resultsCard.hidden = true;
    return;
  }

  passwords.forEach((pwd, index) => {
    const li = document.createElement("li");
    li.className = "results-item";

    const textSpan = document.createElement("span");
    textSpan.className = "results-item-text";
    textSpan.textContent = pwd;

    const indexSpan = document.createElement("span");
    indexSpan.style.fontSize = "0.75rem";
    indexSpan.style.color = "#6b7280";
    indexSpan.textContent = String(index + 1).padStart(2, "0");

    li.appendChild(textSpan);
    li.appendChild(indexSpan);
    resultsList.appendChild(li);
  });

  resultsCard.hidden = false;
}

function getResultsText() {
  const items = Array.from(resultsList.querySelectorAll(".results-item-text"));
  return items.map((el) => el.textContent || "").join("\n").trim();
}

form.addEventListener("submit", async (event) => {

  // const targetPassword = (auditPasswordInput?.value || "").trim();
  // if (!targetPassword) {
  //   setStatus("Please enter the intended password.", true);
  //   return;
  // }
  event.preventDefault();
    const targetPassword = (auditPasswordInput?.value || "").trim();

  if (!targetPassword) {
    setStatus("Please enter the intended password before generating.", true);
    return;
  }
  setAuditStatus("", "info");

  const context = contextInput.value.trim();
  const numPasswords = parseInt(countInput.value, 10) || 10;
  const seedRaw = seedInput?.value?.trim() || "";
  const seed = seedRaw === "" ? null : Number(seedRaw);
  const min_length = parseInt(minLenInput?.value, 10) || 10;
  const max_length = parseInt(maxLenInput?.value, 10) || 32;
  const require_symbol = Boolean(reqSymbolInput?.checked);
  const require_digit = Boolean(reqDigitInput?.checked);

  setStatus("Generating passwords...", false);
  setLoading(true);

  try {
    const response = await fetch("/api/generate-passwords", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        context,
        num_passwords: numPasswords,
        mode: "auto",
        seed,
        min_length,
        max_length,
        require_symbol,
        require_digit,
      }),
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail = payload?.detail || "Generation failed.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    renderResults(payload.passwords || []);
    const items = Array.from(
      resultsList.querySelectorAll(".results-item-text")
    ).map(el => el.textContent.trim());
    
    const idx = items.findIndex(p => p === targetPassword.trim());
    
    const count = items.length;
    
    const resultBox = document.getElementById("audit-status");
    
    if (idx >= 0) {
      resultBox.textContent = `Cracked in ${idx + 1} tries (out of ${count})`;
      resultBox.className = "status-text status-text--success";
    } else {
      resultBox.textContent = `Not cracked within ${count} attempts`;
      resultBox.className = "status-text status-text--error";
    }

    setAuditStatus(
      idx >= 0
        ? `Cracked in ${idx + 1} tries (out of ${count})`
        : `Not cracked within ${count} attempts`,
      idx >= 0 ? "success" : "info"
    );
} catch (error) {
  console.error(error);
  setStatus(error.message || "Something went wrong while generating passwords.", true);
} finally {
  setLoading(false);
}
});

copyAllButton.addEventListener("click", async () => {
  const text = getResultsText();
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("All passwords copied to clipboard.");
  } catch {
    setStatus("Unable to copy to clipboard in this browser.", true);
  }
});

downloadButton?.addEventListener("click", () => {
  const text = getResultsText();
  if (!text) return;

  const blob = new Blob([text + "\n"], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "wordlist.txt";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  setStatus("Downloaded wordlist.txt");
});

// auditButton?.addEventListener("click", () => {
//   const target = (auditPasswordInput?.value || "").trim();
//   if (!target) {
//     setAuditStatus("Enter a password to check coverage.", "error");
//     return;
//   }
//   const items = Array.from(resultsList.querySelectorAll(".results-item-text")).map((el) => el.textContent || "");
//   const idx = items.findIndex((p) => p === target);
//   if (idx >= 0) {
//     setAuditStatus(`Coverage hit: found at rank ${idx + 1}.`, "success");
//   } else {
//     setAuditStatus("Not found in current list. Try generating more or refining context.", "info");
//   }
// });

pdfForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const pwd = (pdfPassword?.value || "").trim();
  const filename = (pdfFilename?.value || "demo_protected.pdf").trim();
  const content = (pdfContent?.value || "").trim();

  if (!pwd) {
    setPdfStatus("Please enter a password to set on the PDF.", true);
    return;
  }

  setPdfStatus("Creating PDF...", false);
  try {
    const res = await fetch("/api/create-demo-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password: pwd, filename, content }),
    });

    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      const detail = payload?.detail || "PDF creation failed.";
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename.toLowerCase().endsWith(".pdf") ? filename : `${filename}.pdf`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);

    setPdfStatus("PDF downloaded.");
  } catch (e) {
    console.error(e);
    setPdfStatus(e.message || "Failed to create PDF.", true);
  }
});

