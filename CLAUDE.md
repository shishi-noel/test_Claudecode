# CLAUDE.md - 離職予測 AI Web アプリ

## プロジェクト概要
Flask製の従業員離職リスク予測Webアプリ。GradientBoostingClassifierモデルを使用し、従業員情報から離職確率を算出する。

## 技術スタック
- **Backend**: Python / Flask
- **ML**: scikit-learn (GradientBoostingClassifier + StandardScaler Pipeline)
- **Model Storage**: joblib (`turnover_model.pkl`)

## ファイル構成
```
app.py              # Flask Webサーバー (ルーティング・予測処理)
turnover_model.py   # モデル定義・学習・推論ロジック
train.py            # モデル学習スクリプト
predict.py          # CLI予測スクリプト
templates/          # HTMLテンプレート
requirements.txt    # 依存パッケージ
```

## 入力特徴量
| 特徴量 | 型 | 説明 |
|--------|-----|------|
| 年齢 | int | 22〜65歳 |
| 性別 | int | 0=男性, 1=女性 |
| 勤務年数 | int | 0〜30年 |
| 夜勤回数 | int | 月あたり0〜20回 |
| ストレス指標 | float | 1.0〜10.0 |

## リスク判定基準
- **高リスク** (≥70%): 離職確率70%以上
- **中リスク** (40〜70%): 離職確率40〜70%
- **低リスク** (<40%): 離職確率40%未満

## セットアップ・実行
```bash
pip install -r requirements.txt
python train.py          # モデル学習（初回のみ）
python app.py            # サーバー起動 → http://localhost:5000
```

## 開発メモ
- 勤務年数が長いほど離職リスクが上がる（`logit`の`0.08 * tenure`項）
- モデルファイル `turnover_model.pkl` は `.gitignore` 対象
- 学習データは `generate_training_data()` で合成生成（5000サンプル）

## Obsidian連携
このCLAUDE.mdはClaude CodeのPostToolUseフック（GitHub MCP経由）によって、GitHubリポジトリに自動同期されます。Obsidian側では **Obsidian Git** プラグインでpullすることで最新のドキュメントを参照できます。
