/* ============================================================
 * Evidra Frontend App Script (app.js)
 * - 観測データ/PDFのアップロード
 * - 解析Runの作成・ポーリング
 * - Step1/Step3 Mermaid描画
 * - Step2 評価表（Markdownテーブル→HTML）
 * - Step4 チャット（Azure OpenAI 連携／履歴表示）
 * 
 * 画面側の想定ID:
 *  - #dataFile, #pdfFile … ファイル入力（観測データ / RAG用PDF）
 *  - #startBtn … 「解析開始」ボタン
 *  - #statusLabel, #progressBar … ステータス表示
 *  - #mermaid1, #mermaid3 … Mermaid出力先（<div>や<pre>どちらでもOK）
 *  - #markdown2 … Step2の評価表（<div>）
 *  - #plotlyLink … Plotly HTMLリンク（<a>）
 *  - #chatLog, #chatInput, #chatSend, #chatClear … チャットUI
 * 
 * 必須ライブラリ:
 *  - Mermaid v11（window.mermaid が利用可能）
 * ============================================================ */

(() => {
  // グローバル状態（Run/アップロード結果を保持）
  window.currentRunId = null;
  let currentDatasetId = null;
  let currentRagDocId = null;

  // Mermaidは自動起動せず手動レンダリングにする（安全性向上）
  if (window.mermaid && typeof window.mermaid.initialize === 'function') {
    try {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'loose',
        theme: 'default'
      });
    } catch (e) {
      console.warn('Mermaid initialize skipped:', e);
    }
  }

  /* ------------------------------
   * ユーティリティ
   * ------------------------------ */

  // JSONフェッチの薄いラッパー（エラー時は例外投げる）
  async function fetchJSON(url, opts = {}) {
    const res = await fetch(url, {
      headers: {'Content-Type': 'application/json'},
      ...opts
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status} ${res.statusText} ${txt}`);
    }
    return res.json();
  }

  // ファイルPOST（FormDataで送る）
  async function uploadFile(url, file, fieldName = 'file') {
    const fd = new FormData();
    fd.append(fieldName, file, file.name);
    const res = await fetch(url, { method: 'POST', body: fd });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`Upload failed: ${res.status} ${res.statusText} ${txt}`);
    }
    return res.json();
  }

  // ステータス/プログレスの表示更新
function updateStatusUI(label, pct, stageStatuses) {
  const labelElm = document.getElementById('statusLabel');
  if (labelElm) labelElm.textContent = label || '未実行';
  const bar = document.getElementById('progressBar');
  if (bar) {
    const v = Math.min(100, Math.max(0, pct || 0));
    bar.style.width = v + '%';
    bar.setAttribute('aria-valuenow', String(v));
  }
  const ss = stageStatuses || {};
  const map = { state1: 'Step1', state2: 'Step2', state3: 'Step3' };
  Object.entries(map).forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (el) el.textContent = ss[key] || '未実行';
  });
}

  // Mermaid描画（textコード→SVGにレンダリング）
  async function renderMermaid(elId, code) {
    const el = document.getElementById(elId);
    if (!el) return;
    // 完全に上書き
    el.innerHTML = '';
    el.removeAttribute('data-processed');   // ← v11 対策：再レンダ抑止フラグを外す
    el.textContent = code || '';
    try {
      await mermaid.run({ nodes: [el] });
    } catch (e) {
      console.warn('Mermaid render error:', e);
    }
  }

  // Markdown表（GitHub風 | 区切り）→ HTML <table> に変換
  function renderMarkdownTable(elmId, md) {
    const container = document.getElementById(elmId);
    if (!container) return;
    container.innerHTML = '';
    if (!md || !md.trim()) return;

    const lines = md.trim().split('\n').filter(l => l.trim().length > 0);
    if (lines.length < 2) {
      container.textContent = md;
      return;
    }
    const headerLine = lines[0];
    const sepLine = lines[1];

    const splitRow = (row) =>
      row.split('|').map(s => s.trim())
         .filter((_, i, arr) => !(i === 0 && arr[0] === '') && !(i === arr.length - 1 && arr[arr.length - 1] === ''));

    const headers = splitRow(headerLine);
    if (!sepLine.includes('---')) { container.textContent = md; return; }

    const table = document.createElement('table');
    table.className = 'table';

    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    headers.forEach(h => {
      const th = document.createElement('th');
      th.textContent = h;
      trh.appendChild(th);
    });
    thead.appendChild(trh);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    for (let i = 2; i < lines.length; i++) {
      const cols = splitRow(lines[i]);
      if (cols.length === 0) continue;
      const tr = document.createElement('tr');
      cols.forEach((c, idx) => {
        const td = document.createElement('td');
        if (headers[idx] && headers[idx].toLowerCase().includes('citations') && c !== '-' && c.length > 0) {
          const parts = c.split(',').map(s => s.trim());
          parts.forEach((p, j) => {
            const btn = document.createElement('button');
            btn.textContent = p;
            btn.className = 'cite-btn';
            btn.onclick = () => {
              // 実運用では Cosmos から該当chunkテキストを取得して 10–20 行程度をモーダルで表示する
              alert('スニペット（ダミー表示）\n\n' + p + '\n\n※実接続時はCosmosから該当chunkを取得して表示します。');
            };
            td.appendChild(btn);
            if (j < parts.length - 1) td.appendChild(document.createTextNode(' '));
          });
        } else {
          td.textContent = c;
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
  }

  // チャットログを描画（roleに応じてCSSクラス付与）
  function renderChatLog(msgs) {
    const box = document.getElementById('chatLog');
    if (!box) return;
    box.innerHTML = '';
    (msgs || []).forEach(m => {
      const div = document.createElement('div');
      div.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant');
      div.textContent = m.text;
      box.appendChild(div);
    });
    box.scrollTop = box.scrollHeight;
  }

  // 会話履歴をAPIから取得して描画
  async function loadChatHistory() {
    if (!window.currentRunId) return;
    try {
      const data = await fetchJSON(`/api/chat/${window.currentRunId}/history`);
      renderChatLog(data.messages || []);
    } catch (e) {
      console.warn('loadChatHistory failed:', e);
    }
  }

  /* ------------------------------
   * アップロード: 観測データ / RAG PDF
   * ------------------------------ */

  // アップロード直後のプレビュー表を描画
  function renderPreviewTable(elmId, columns, rows) {
    const box = document.getElementById(elmId);
    if (!box) return;
    box.innerHTML = '';
    if (!rows || rows.length === 0) { box.textContent = '(プレビューなし)'; return; }
    const table = document.createElement('table');
    table.className = 'table';
    const thead = document.createElement('thead');
    const trh = document.createElement('tr');
    (columns || Object.keys(rows[0] || {})).forEach(col => {
      const th = document.createElement('th'); th.textContent = col; trh.appendChild(th);
    });
    thead.appendChild(trh); table.appendChild(thead);
    const tbody = document.createElement('tbody');
    rows.forEach(r => {
      const tr = document.createElement('tr');
      (columns || Object.keys(rows[0] || {})).forEach(col => {
        const td = document.createElement('td'); td.textContent = (r[col] ?? ''); tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    box.appendChild(table);
  }

  // 観測データ（CSV/XLSX）アップロード → dataset_id取得 → 「解析開始」活性化
  document.getElementById('dataFile')?.addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    try {
      const json = await uploadFile('/api/upload-data', file, 'file');
      currentDatasetId = json.dataset_id;
      // 観測データがあれば解析開始を有効化（RAGは任意）
      const startBtn = document.getElementById('startBtn');
      if (startBtn) startBtn.disabled = !currentDatasetId;
      // プレビューを即表示（#dataPreview がテンプレ内にある前提）
      renderPreviewTable('dataPreview', json.columns || [], json.head_preview || []);
    } catch (e) {
      alert('観測データのアップロードに失敗しました。\n' + e.message);
      console.error(e);
    }
  });

  // PDF（任意）アップロード → rag_doc_id取得（RAGなしでも解析は可能）
  document.getElementById('pdfFile')?.addEventListener('change', async (ev) => {
    const file = ev.target.files?.[0];
    if (!file) return;
    try {
      const json = await uploadFile('/api/upload-pdf', file, 'file');
      currentRagDocId = json.rag_doc_id;
      console.log('RAG doc uploaded:', json);
    } catch (e) {
      alert('RAG PDFのアップロードに失敗しました。\n' + e.message);
      console.error(e);
    }
  });

  /* ------------------------------
   * Runの作成・ポーリング・成果物取得
   * ------------------------------ */

  // 解析開始
  document.getElementById('startBtn')?.addEventListener('click', async () => {
    if (!currentDatasetId) {
      alert('観測データが未アップロードです。アップロード後に開始してください。');
      return;
    }
    // 解析パラメータ（UIがあればそこから取得。ここでは既定値を例示）
    const params = {
      lag: parseInt(document.getElementById('param_lag').value, 10),
      boot: parseInt(document.getElementById('param_boot').value, 10),
      seed: 42,
      preprocessing: { standardize: true },
      edge_threshold: parseFloat(document.getElementById('param_thr').value) || 1e-10
    };
    try {
      const payload = {
        dataset_id: currentDatasetId,
        rag_doc_id: currentRagDocId || null,
        params
      };
      const json = await fetchJSON('/api/run', { method: 'POST', body: JSON.stringify(payload) });
      window.currentRunId = json.run_id;
      // ステータス初期化
      updateStatusUI('ジョブ起動', 1, {'Step1': '処理中'});
      // 履歴ロード（Step4表示のため）
      loadChatHistory();
      // ポーリング開始
      startPollingStatus(window.currentRunId);
    } catch (e) {
      alert('Runの作成に失敗しました。\n' + e.message);
      console.error(e);
    }
  });

  // ステータスポーリング（2秒間隔で /status → 完了で /artifacts 取得）
  let pollTimer = null;
  function startPollingStatus(runId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      try {
        const st = await fetchJSON(`/api/run/${runId}/status`);
        const s = st.status || {};
        updateStatusUI(s.label || '', s.pct || 0, s.stage_statuses || {});
        // 完了/失敗/キャンセルのいずれかで停止
        const overall = (s.overall || '').toLowerCase();
        if (['succeeded','failed','cancelled'].includes(overall)) {
          clearInterval(pollTimer);
          pollTimer = null;
          // 成果物取得
          loadArtifacts(runId);
        }
      } catch (e) {
        console.warn('poll status failed:', e);
      }
    }, 2000);
  }

  // 成果物取得 → Mermaid/表/Plotlyリンクを描画
  async function loadArtifacts(runId) {
    try {
      const art = await fetchJSON(`/api/run/${runId}/artifacts`);
      // Step1 Mermaid
      // if (art.mermaid_step1) await renderMermaid('mermaid1', art.mermaid_step1);
      // Step2 評価表
      if (art.markdown_table) {
        renderMarkdownTable('markdown2', art.markdown_table);
      }
      const lg = document.getElementById('typeLegend');
      if (lg) {
        lg.innerHTML = `
          <strong>TYPE 凡例</strong>
          <ul>
            <li>TYPE1：因果なし</li>
            <li>TYPE2：因果あり・因果の向き同じ・因果の正負同じ</li>
            <li>TYPE3：因果あり・因果の向き同じ・因果の正負違う</li>
            <li>TYPE4：因果あり・因果の向き違う・因果の正負同じ</li>
            <li>TYPE5：因果あり・因果の向き違う・因果の正負違う</li>
          </ul>`;
      }
      // Step3 Mermaid
      if (art.mermaid_step3) await renderMermaid('mermaid3', art.mermaid_step3);
      // // Plotly HTML（公開URL: /media/plots/xxx.html 等）
      // const a = document.getElementById('plotlyLink');
      // if (a) {
      //   if (art.plotly_html_path && typeof art.plotly_html_path === 'string') {
      //     a.href = art.plotly_html_path;
      //     a.target = '_blank';
      //     a.textContent = 'Plotly図を開く';
      //     a.style.display = '';
      //   } else {
      //     a.removeAttribute('href');
      //     a.style.display = 'none';
      //   }
      // }
    } catch (e) {
      console.warn('loadArtifacts failed:', e);
    }
  }

  /* ------------------------------
   * Step4 チャット
   * ------------------------------ */

  // 送信 → サーバ応答を追記 → 最後に履歴APIで再描画（保存内容で正とする）
  document.getElementById('chatSend')?.addEventListener('click', async () => {
    const inp = document.getElementById('chatInput');
    const text = (inp.value || '').trim();
    if (!text) return;
    if (!window.currentRunId) {
      alert('Runが存在しません。解析を実行してからご利用ください。');
      return;
    }
    // 楽観的に自分の発言を描画
    const log = document.getElementById('chatLog');
    if (log) {
      const me = document.createElement('div');
      me.className = 'msg user';
      me.textContent = text;
      log.appendChild(me);
      log.scrollTop = log.scrollHeight;
    }
    inp.value = '';

    try {
      const res = await fetch(`/api/chat/${window.currentRunId}`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      // 一旦レスポンスを追記
      if (log) {
        const bot = document.createElement('div');
        bot.className = 'msg assistant';
        bot.textContent = data.answer || '(no answer)';
        log.appendChild(bot);
        log.scrollTop = log.scrollHeight;
      }
      // 最終的にはサーバ保存の履歴で上書き再描画（消えてしまう問題の対策）
      await loadChatHistory();
    } catch (e) {
      if (log) {
        const err = document.createElement('div');
        err.className = 'msg assistant';
        err.textContent = '送信に失敗しました';
        log.appendChild(err);
      }
      console.error(e);
    }
  });

  // Enterキーで送信（Shift+Enterは改行）
  document.getElementById('chatInput')?.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      document.getElementById('chatSend')?.click();
    }
  });

  // 履歴クリア（UIのみソフトクリア。サーバ側の保存は保持）
  document.getElementById('chatClear')?.addEventListener('click', () => {
    const log = document.getElementById('chatLog');
    if (log) log.innerHTML = '';
  });

  // ページロード時に、（もし currentRunId が残っていれば）履歴を読み直す
  window.addEventListener('load', () => {
    // 初回は解析未実行の想定だが、必要ならここで既存Run IDを復元して履歴を取る
    loadChatHistory();
    // 解析開始ボタンは観測データが無い限り押せないようにする
    const startBtn = document.getElementById('startBtn');
    if (startBtn) startBtn.disabled = !currentDatasetId;
    // Plotlyリンクは初期は隠す
    const a = document.getElementById('plotlyLink');
    if (a) a.style.display = 'none';
  });
})();
