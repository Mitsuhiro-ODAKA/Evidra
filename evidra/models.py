from django.db import models

# 注意: 本番はCosmos DBに寄せるが、開発中はSQLiteで同等スキーマを保持する。
# 将来は独自リポジトリ層を挟んでストレージ切り替えに対応する。

class Dataset(models.Model):
    # 観測データのメタ情報を保持する
    created_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=1024)         # ローカル保存先（本番はBlob URL）
    columns_json = models.JSONField(default=list)         # 列名リスト
    n_rows = models.IntegerField(default=0)
    freq_guess = models.CharField(max_length=64, blank=True, default="")  # 周波数推定（年/四半期/月など）

class RagDoc(models.Model):
    # RAG用PDFのメタ情報を保持する
    created_at = models.DateTimeField(auto_now_add=True)
    file_path = models.CharField(max_length=1024)         # ローカル保存先（本番はBlob URL）
    pages = models.IntegerField(default=0)
    size_mb = models.FloatField(default=0.0)

class Run(models.Model):
    # 実行単位（再現実行/replayで同じ設定を再利用する）
    created_at = models.DateTimeField(auto_now_add=True)
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    rag_doc = models.ForeignKey(RagDoc, on_delete=models.SET_NULL, null=True, blank=True)

    # パラメータ（ラグ、ブート回数、seed、前処理設定）
    params = models.JSONField(default=dict)

    # ステータス（overall/step/pct/cancel_requested 等）
    status = models.JSONField(default=dict)

    # タイムアウトや警告類
    warnings = models.JSONField(default=list)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

class Edge(models.Model):
    # エッジリスト（因果発見結果および妥当性評価結果）
    run = models.OneToOneField(
        Run,
        on_delete=models.CASCADE,
        related_name='artifact',   # Run.artifact で1件にアクセス
    )
    source = models.CharField(max_length=128)
    target = models.CharField(max_length=128)
    effect = models.FloatField(default=0.0)                # 係数推定値（標準化表示はしない）
    prob = models.FloatField(default=0.0)                  # ブートストラップ出現頻度 (0..1)
    sign = models.CharField(max_length=1, default='+')     # '+' or '-'

    # 妥当性評価
    eval_has = models.BooleanField(null=True)              # 因果の有無
    eval_dir = models.BooleanField(null=True)              # 向きが同じか
    eval_sign = models.BooleanField(null=True)             # 正負が同じか
    type_code = models.IntegerField(null=True)             # TYPE 1..5

    # RAG引用ID
    citations = models.JSONField(default=list)             # [{doc_id, page, snippet_id}, ...]

class Artifact(models.Model):
    # 表示/ダウンロード用成果物（MermaidやPlotlyのリンクなど）
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name='artifacts')
    mermaid_step1 = models.TextField(blank=True, default="")
    mermaid_step3 = models.TextField(blank=True, default="")
    markdown_table = models.TextField(blank=True, default="")
    plotly_html_path = models.CharField(max_length=1024, blank=True, default="")  # ローカル保存（本番はBlob URL）

class Chat(models.Model):
    # チャット履歴（Runに紐づけ、クリア時はソフト削除）
    run = models.ForeignKey(Run, on_delete=models.CASCADE, related_name='chats')
    created_at = models.DateTimeField(auto_now_add=True)
    role = models.CharField(max_length=16)     # 'user' or 'assistant' 等
    text = models.TextField()
    soft_deleted = models.BooleanField(default=False)
