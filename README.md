# ACTA
Optimization Framework of Task Allocation in Extreme Environments with Asymmetric-Communication Condition

uv run python scripts/run_sim_once.py --scenario configs/scenarios/toy_scenario_001.yml

uv run solara run scripts/acta_viz.py


全ワーカーの情報を受け取る（拠点・近隣ワーカー）
全ワーカーの情報を更新
タスクの１ステップ仕事量を初期化
全ワーカーの行動を決定
	ルートの最適化
		
	既に「修理に行く途中」「修理中」なら行動は変えない
	ローカル情報をもとにルートの中から次のタスクを選択
	もし全ルート終了していたらその場で待機
	RepairFlagsを確認
		「フラグが立っている & まだその仕事では repair していない」時だけ修理開始
		そのフラグで修理したことを記録
全ワーカー行動
	故障判定
	修理中
	修理に移動中
	仕事
タスクの仕事量を更新


TODO:
データコレクター
    履歴を取る
    AoI
    

自律分散

可視化ツール
    通信範囲
    故障状態
