'use strict';
// UI language. Strings are written in Japanese in the HTML and the scripts; this file holds their English
// versions and swaps them at start-up (static text) or at call time (t()). Language: TSC_CONFIG.lang
// ('ja' in the development app, 'en' in the published BrowSeg build) or ?lang=en|ja.
const LANG = (() => {
  const q = new URLSearchParams(location.search).get('lang');
  return q || (self.TSC_CONFIG && TSC_CONFIG.lang) || 'ja';
})();

const I18N_EN = {
  // header / tabs
  'TSC++ Annotator': 'BrowSeg',
  'TotalSegmentator (C++ / WebAssembly / WebGPU) による自動アノテーション・修正・学習データ作成 — 画像はこの PC の外に送信されません':
    'TotalSegmentator (C++ / WebAssembly / WebGPU) for auto-annotation, correction and training data — images never leave this PC',
  'TotalSegmentator (C++ / WebAssembly / WebGPU) をブラウザ内で実行 — 画像はこの PC の外に送信されません':
    'TotalSegmentator (C++ / WebAssembly / WebGPU) running inside the browser — images never leave this PC',
  'エンジン起動中…': 'Starting engine…', 'バックエンド確認中…': 'Checking back end…',
  'データ': 'Data', '解析': 'Analyze', 'ラベル': 'Labels',
  // data tab
  'CT を開く': 'Open a CT', 'DICOM フォルダをドロップ': 'Drop a DICOM folder', 'またはクリックして選択': 'or click to choose',
  'NIfTI (.nii/.nii.gz) を開く': 'Open NIfTI (.nii/.nii.gz)', '別のシリーズを選ぶ': 'Choose another series',
  'ラベルを読み込む': 'Load labels', 'ラベル NIfTI を読み込む': 'Load a label NIfTI',
  '3D Slicer / ITK-SNAP 等で編集したラベルも読み込めます（同じ CT の格子であること）。meta.json を一緒に選ぶとラベル名も復元します。':
    'Labels edited in 3D Slicer, ITK-SNAP etc. can be loaded too (same voxel grid as the CT). Select the meta.json as well to restore the label names.',
  '保存済み症例（バックエンド）': 'Saved cases (back end)', '書き出し': 'Export',
  'ラベル .nii.gz': 'Labels .nii.gz', 'メッシュ OBJ': 'Mesh OBJ', 'メッシュ STL': 'Mesh STL',
  // analyze tab
  '自動アノテーション': 'Segmentation', '既存ラベルに統合': 'Merge into existing labels', '置き換え': 'Replace', '実行': 'Run',
  '統合: 新しい結果のラベルが既存の同じ場所を上書きします（ラベル名が同じなら同じラベルになります）。実行は「元に戻す」で取り消せます。':
    'Merge: the new labels overwrite existing labels at the same voxels (a label with the same name stays the same label). A run can be undone.',
  'ログ': 'Log',
  // labels tab
  '全表示': 'Show all', '全非表示': 'Hide all', '新しいラベル': 'New label', '追加': 'Add', '選択中のラベル': 'Selected label',
  'なし': 'none', '確認済み（学習に使う）': 'Verified (use for training)', '3D 不透明度': '3D opacity',
  '名前変更': 'Rename', 'ラベル削除（ボクセルも消去）': 'Delete label (clears its voxels)',
  'ラベルを検索': 'Search labels', 'すべて表示': 'Show all', 'すべて非表示': 'Hide all',
  '名前 (例: portal_vein_branch)': 'Name (e.g. portal_vein_branch)', '新しい名前': 'New name',
  // toolbar
  '✥ 移動': '✥ Move', '🖌 ブラシ': '🖌 Brush', '⌫ 消しゴム': '⌫ Eraser', '▣ 塗りつぶし': '▣ Fill', '⇄ 領域付け替え': '⇄ Relabel region',
  '⊙ スポイト': '⊙ Pick', '半径': 'Radius', '球': 'Sphere', '他ラベル保護': 'Protect other labels', '全ラベル消去': 'Erase all labels',
  'CT値': 'HU range', '腹部': 'Abdomen', '肝臓': 'Liver', '骨': 'Bone', '肺': 'Lung', '脳': 'Brain', '造影血管': 'Contrast vessels',
  '十字線': 'Crosshair', '3D 更新': 'Update 3D', '自動': 'auto',
  '移動・十字線 (N)': 'Move / crosshair (N)', 'ブラシ (B)': 'Brush (B)', '消しゴム (E)': 'Eraser (E)', '2D 塗りつぶし (F)': '2D fill (F)',
  '3D 連結領域の付け替え (R)': 'Relabel a 3D connected region (R)', 'ラベルを拾う (I / Alt+クリック)': 'Pick a label (I / Alt+click)',
  '球状ブラシ（断面をまたいで塗る）': 'Spherical brush (paints across slices)', '他のラベルを上書きしない': 'Do not overwrite other labels',
  '消しゴムで全ラベルを消す': 'Eraser removes every label', 'この CT 値の範囲だけ編集する': 'Edit only within this HU range',
  '元に戻す (Ctrl+Z)': 'Undo (Ctrl+Z)', 'やり直し (Ctrl+Y)': 'Redo (Ctrl+Y)', 'ウィンドウ': 'Window', '重ね表示の濃さ': 'Overlay opacity',
  '編集したラベルの 3D を更新': 'Update the 3D view of edited labels',
  // views
  '3D（ドラッグ: 回転 / 右ドラッグ: 移動 / ホイール: 拡大 / ダブルクリック: 全体表示）': '3D (drag: rotate / right-drag: pan / wheel: zoom / double-click: fit)',
  'DICOM フォルダをドロップするか、「データ」タブから CT を開いてください': 'Drop a DICOM folder here, or open a CT from the Data tab',
  // series dialog
  'DICOM シリーズの選択': 'Choose a DICOM series', '説明': 'Description', '種類': 'Modality', '枚数': 'Slices', 'マトリクス': 'Matrix',
  '画素 (mm)': 'Pixel (mm)', 'スライス間隔 (mm)': 'Slice spacing (mm)', '範囲 (mm)': 'Extent (mm)', '時刻': 'Time', '造影剤': 'Contrast',
  '備考': 'Notes', 'キャンセル': 'Cancel', '読み込む': 'Load',
  // main.js
  '肝臓 (total, roi_subset=liver)': 'Liver (total, roi_subset=liver)', '肝臓 8 区域 (Couinaud)': 'Liver segments (Couinaud, 8)',
  '肝内血管 (統合)': 'Liver vessels', '腹部主要臓器 (roi_subset)': 'Main abdominal organs (roi_subset)',
  '全身 117 構造 (total, 5 モデル)': 'All 117 structures (total, 5 models)',
  '体幹の腔 (trunk_cavities: 腹腔・胸腔・心膜・縦隔)': 'Trunk cavities (trunk_cavities: abdominal, thoracic, pericardium, mediastinum)',
  '体 (body: 体幹・四肢)': 'Body (body: trunk, extremities)', '体 6 mm (body --fast)': 'Body 6 mm (body --fast)',
  'エンジンが停止しています。ページを再読み込みしてください': 'The engine has stopped. Reload the page.',
  'エンジン {0}（GPU が使えなくなったため切替）': 'Engine {0} (switched: the GPU became unavailable)',
  'エンジンが停止しました: {0}（ページを再読み込みしてください）': 'The engine stopped: {0} (reload the page)',
  'エンジン停止': 'Engine stopped', 'エンジン {0} / {1} スレッド{2}': 'Engine {0} / {1} threads{2}', '（単一スレッド版）': ' (single-thread build)',
  'エンジン起動失敗': 'Engine failed to start', 'エンジンの起動に失敗しました: {0}': 'The engine failed to start: {0}',
  'DICOM → ボリューム': 'DICOM → volume', '6mm へリサンプル': 'Resample to 6 mm', '粗セグメンテーション': 'Coarse segmentation',
  'リサンプル': 'Resample', 'モデル推論': 'Network', '完了': 'Done', 'メッシュ生成': 'Meshes', 'DICOM 読み込み': 'Reading DICOM',
  '背景': 'background', 'シリーズ': 'series',
  'DICOM ファイルが見つかりません': 'No DICOM files found', '処理中です。終わってから読み込んでください': 'Busy; load after the current task has finished',
  'DICOM 読み込み中…': 'Reading DICOM…', '2 枚以上の画像からなるシリーズがありません': 'No series with two or more images',
  'エンジン {0}': 'Engine {0}', '読み込み失敗: {0}': 'Loading failed: {0}', '（表示中の CT も閉じました）': ' (the displayed CT was closed as well)',
  '門脈相（肝血管の抽出向き）': 'portal venous phase (best for liver vessels)', '枚数が最も多い CT シリーズ': 'CT series with the most slices',
  '同じ位置のスライスが重複（複数の相が 1 シリーズに混在？）・読み込めません': 'duplicate slice positions (several phases in one series?) — cannot be loaded',
  '向きの違うスライスが混在・読み込めません': 'slices with different orientations — cannot be loaded',
  '位置決め画像のみ（読み込めません）': 'localizer images only (cannot be loaded)', '画像が 1 枚のみ（読み込めません）': 'only one image (cannot be loaded)',
  '位置決め画像など {0} 枚は除外': '{0} images excluded (localizers etc.)', '{0}（CT 以外）': '{0} (not CT)', '厚いスライス': 'thick slices', '、': ', ',
  '(説明なし)': '(no description)', '推奨': 'suggested',
  '{0} シリーズが見つかりました。推奨: {1}（{2}）': '{0} series found. Suggested: {1} ({2})',
  '。切り替えると現在のラベルは消えます（必要なら先に保存してください）。': '. Switching discards the current labels (save them first if needed).',
  'シリーズ読み込み中…': 'Loading series…', 'NIfTI 読み込み中…': 'Reading NIfTI…',
  'ラベル値 {0} は 0〜255 の整数ではありません': 'Label value {0} is not an integer in 0–255', 'ラベル値 {0} は 255 を超えています': 'Label value {0} exceeds 255',
  'ラベル NIfTI を選んでください': 'Choose a label NIfTI', 'ラベルを読み込めません: {0}': 'Cannot load the labels: {0}',
  '格子が CT と違います ({0} vs {1})': 'The grid differs from the CT ({0} vs {1})', 'カスタム: {0} v{1} ({2})': 'Custom: {0} v{1} ({2})',
  '解析中…': 'Segmenting…', '完了 ({0} 秒)': 'Done ({0} s)', '解析失敗: {0}': 'Segmentation failed: {0}',
  '完了: {0}（{1} 秒）': 'Done: {0} ({1} s)', '解析失敗（{0}）: {1}': 'Segmentation failed ({0}): {1}',
  'WebGPU が使えないため CPU で解析します（結果は同じで、時間がかかります）。Linux の Chrome では chrome://flags で「Unsafe WebGPU Support」と「Vulkan」を Enabled にして Chrome を再起動すると GPU を使えます。': 'WebGPU is not available, so segmentation runs on the CPU (same results, slower). In Chrome on Linux, set "Unsafe WebGPU Support" and "Vulkan" to Enabled in chrome://flags and relaunch Chrome to use the GPU.',
  '{0} ラベル   編集 {1} voxels   {2} タスク': '{0} labels   edited {1} voxels   {2} tasks',
  '表示': 'show', '色を変更': 'change colour', '由来: {0}': 'source: {0}', 'なし（一覧でクリックして選択）': 'none (click one in the list)',
  '名前を入力してください': 'Enter a name', 'ラベル「{0}」を追加しました。ブラシで塗ってください': 'Label "{0}" added. Paint it with the brush',
  '新しい名前を入力してください': 'Enter the new name', '同じ名前のラベルがあります': 'A label with that name exists',
  'ラベル「{0}」を削除しました（元に戻せます）': 'Label "{0}" deleted (can be undone)',
  // editor.js
  '処理中は編集できません': 'Editing is disabled while a task is running',
  '先に編集するラベルを選択してください（ラベル一覧でクリック）': 'Select the label to edit first (click it in the label list)',
  '塗りつぶし範囲が断面の半分を超えました。CT値範囲の制限を使うと安全です': 'The fill covered more than half of the slice. Limiting the HU range is safer',
  '背景ではなくラベルのある場所をクリックしてください': 'Click a labelled voxel, not the background',
  'クリックした領域は既に選択中のラベルです': 'The clicked region already has the selected label',
  '{0} ボクセルを付け替えました': 'Relabelled {0} voxels',
  // labels.js
  'ラベル数の上限 (255) です': 'Label limit (255) reached', 'ラベル「{0}」は既にあります': 'Label "{0}" already exists',
  // worker.js
  'DICOM の画像が見つかりません（非圧縮の DICOM が必要です）': 'No DICOM images found (uncompressed DICOM is required)',
  // V3 liver bundle
  '肝臓一式（V3）': 'Liver bundle (V3)',
  '肝臓・Couinaud 区域 S1–S8・門脈・肝静脈・胆嚢・腫瘍候補を自動で求め、手術シミュレータ用の OBJ 一式（下大静脈を含む 14 ファイル）を ZIP で書き出します。':
    'Finds the liver, Couinaud segments S1–S8, portal and hepatic veins, gallbladder and tumour candidates automatically and writes the OBJ set for the surgical simulator (14 files with the inferior vena cava) as a ZIP.',
  '下大静脈を肝静脈のファイルに含める（BLISS 用）': 'Put the inferior vena cava into the hepatic vein file (for BLISS)', '脾静脈を門脈に足す': 'Add the splenic vein to the portal vein',
  '肝臓一式を実行': 'Run liver bundle', 'OBJ 一式 (.zip)': 'OBJ set (.zip)',
  'CT を読み込んだら自動で実行': 'Run automatically when a CT is loaded', '終わったら ZIP を保存': 'Save the ZIP when done',
  '肝臓・胆嚢・下大静脈・門脈本幹': 'Liver, gallbladder, IVC, portal trunk', 'Couinaud 区域 S1–S8': 'Couinaud segments S1–S8',
  '門脈・肝静脈（OpenPVHV モデル 950）': 'Portal and hepatic veins (OpenPVHV model 950)', '腫瘍候補（liver_vessels）': 'Tumour candidates (liver_vessels)',
  '肝臓一式を解析中…': 'Running the liver bundle…', '肝臓一式 {0}/{1}: {2}': 'Liver bundle {0}/{1}: {2}',
  '肝臓一式 完了（{0} 秒）': 'Liver bundle done ({0} s)', '肝臓一式 失敗: {0}': 'Liver bundle failed: {0}',
  // V3 liver bundle: extra organs
  '追加の臓器（任意）': 'Extra organs (optional)',
  '選んだ臓器は、肝臓一式の後に total をもう 1 回流して求め、ZIP の extra/ フォルダに OBJ で入れます。一式のファイルは変わりません。':
    'The chosen organs come from one more total run after the liver bundle and go to the extra/ folder of the ZIP as OBJ files. The files of the bundle do not change.',
  '臓器を検索': 'Search organs', '選択解除': 'Clear', '{0} 個選択': '{0} selected',
  '追加の臓器: {0}': 'Extra organs: {0}', '追加の臓器 失敗: {0}': 'Extra organs failed: {0}',
};

// English display names of TotalSegmentator structures (labels.js has the Japanese ones)
const I18N_EN_LABELS = {
  liver: 'Liver', spleen: 'Spleen', kidney_right: 'Right kidney', kidney_left: 'Left kidney', gallbladder: 'Gallbladder', stomach: 'Stomach',
  pancreas: 'Pancreas', adrenal_gland_right: 'Right adrenal gland', adrenal_gland_left: 'Left adrenal gland', esophagus: 'Esophagus',
  trachea: 'Trachea', small_bowel: 'Small bowel', duodenum: 'Duodenum', colon: 'Colon', urinary_bladder: 'Urinary bladder',
  prostate: 'Prostate', heart: 'Heart', aorta: 'Aorta', inferior_vena_cava: 'Inferior vena cava',
  portal_vein_and_splenic_vein: 'Portal and splenic vein', portal_vein_intrahepatic: 'Intrahepatic portal vein', hepatic_veins: 'Hepatic veins',
  liver_vessels: 'Liver vessels', liver_tumor: 'Liver tumour', lung_upper_lobe_left: 'Left upper lobe', lung_lower_lobe_left: 'Left lower lobe',
  lung_upper_lobe_right: 'Right upper lobe', lung_middle_lobe_right: 'Right middle lobe', lung_lower_lobe_right: 'Right lower lobe',
  spinal_cord: 'Spinal cord', sacrum: 'Sacrum', sternum: 'Sternum', costal_cartilages: 'Costal cartilages', brain: 'Brain', skull: 'Skull',
};

// t('日本語 {0}', a) -> translated (or the Japanese key) with {n} replaced
function t(key, ...args) {
  let s = LANG === 'en' && Object.prototype.hasOwnProperty.call(I18N_EN, key) ? I18N_EN[key] : key;
  for (let i = 0; i < args.length; i++) s = s.split(`{${i}}`).join(String(args[i]));
  return s;
}

// static text of the page: text nodes, title / placeholder attributes, <title>
function applyI18nToDocument() {
  if (typeof document === 'undefined') return;
  document.documentElement.lang = LANG;
  if (LANG !== 'en') return;
  const swap = (s) => {
    const k = s.trim();
    if (!k || !Object.prototype.hasOwnProperty.call(I18N_EN, k)) return s;
    return s.replace(k, I18N_EN[k]);
  };
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  for (let n = walker.nextNode(); n; n = walker.nextNode()) if (/[぀-ヿ一-鿿]/.test(n.nodeValue)) nodes.push(n);
  for (const n of nodes) n.nodeValue = swap(n.nodeValue);
  for (const el of document.querySelectorAll('[title],[placeholder]')) {
    for (const a of ['title', 'placeholder']) if (el.hasAttribute(a)) el.setAttribute(a, swap(el.getAttribute(a)));
  }
  document.title = swap(document.title);
}

if (typeof self !== 'undefined') { self.t = t; self.LANG = LANG; self.I18N_EN_LABELS = I18N_EN_LABELS; }
