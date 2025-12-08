# scripts/acta_viz.py
from __future__ import annotations

from pathlib import Path
from typing import List

import solara
from mesa.visualization import SolaraViz, make_space_component
from mesa.visualization.components import AgentPortrayalStyle

from acta.sim.scenario_loader import load_scenario_config
from acta.sim.model import ACTAScenarioModel
from acta.sim.agent import WorkerAgent, TaskAgent, CommanderAgent


# プロジェクトルート（scripts/ の 1つ上）
ROOT_DIR = Path(__file__).resolve().parent.parent
SCENARIO_DIR = ROOT_DIR / "configs" / "scenarios"


class ACTAVisModel(ACTAScenarioModel):
    """SolaraViz 用のラッパーモデル."""

    def __init__(self, scenario_path: str):
        cfg = load_scenario_config(scenario_path)
        super().__init__(cfg=cfg)


# --- エージェントの描画スタイル ---
def agent_portrayal(agent):
    # Worker: 青い丸
    if isinstance(agent, WorkerAgent):
        return AgentPortrayalStyle(color="tab:blue", size=50)

    # Task: 状態で色を変える
    if isinstance(agent, TaskAgent):
        if agent.status == "pending":
            color = "tab:gray"
        elif agent.status == "in_progress":
            color = "tab:orange"
        else:  # done
            color = "tab:green"
        return AgentPortrayalStyle(color=color, size=40, marker="s")

    # CommandCenter: 赤いダイヤ
    if isinstance(agent, CommanderAgent):
        return AgentPortrayalStyle(color="tab:red", size=60, marker="D")

    # その他（今はたぶん出てこない）
    return AgentPortrayalStyle(color="black", size=30)


# 連続空間コンポーネントは共通で 1 個だけ作って使い回す
SPACE_COMPONENT = make_space_component(agent_portrayal)


def list_scenarios() -> List[Path]:
    """configs/scenarios/*.yml を列挙して返す。"""
    if not SCENARIO_DIR.exists():
        return []
    return sorted(SCENARIO_DIR.glob("*.yml"))


@solara.component
def Page():
    """Solara が拾うエントリポイント。

    - 上部にシナリオ選択のドロップダウン
    - 下に Mesa の可視化
    """
    scenarios = list_scenarios()  # List[Path]

    # Hooks のルール違反にならないよう、早期 return より前に use_state を呼ぶ
    default_label = scenarios[0].name if scenarios else ""
    selected_label, set_selected_label = solara.use_state(default_label)  # noqa: SH101

    # シナリオが 1 個もないときはメッセージだけ出す
    if not scenarios:
        return solara.Markdown(
            f"**No scenario files found in `{SCENARIO_DIR}`**  \n"
            "configs/scenarios/ に *.yml を置いてください。"
        )

    # ラベル文字列 -> フルパス のマップを作る
    scenario_labels = [p.name for p in scenarios]
    scenario_map = {p.name: str(p) for p in scenarios}

    # 現在選択されているラベルに対応するパスを取得
    if selected_label not in scenario_map:
        # 何かの拍子に変な値になっていたら先頭へフォールバック
        selected_label = scenario_labels[0]
        # state 自体の修正は次のレンダーで行われるのでここでは set_selected_label は呼ばなくてOK

    selected_path = scenario_map[selected_label]

    # ★ ここがポイント：インスタンスを作って渡す
    model = ACTAVisModel(selected_path)

    # コンストラクタ引数 scenario_path を model_params にも渡しておく
    # （Reset ボタン等で再生成するときに使われる）
    viz = SolaraViz(
        model,
        components=[SPACE_COMPONENT],
        model_params={"scenario_path": selected_path},
        name="ACTA Toy Scenario",
    )

    # 上部のコントロール部分
    controls = solara.Column(
        children=[
            solara.Markdown("## ACTA Scenario Viewer"),
            solara.Select(
                label="Scenario YAML",
                value=selected_label,      # ← ラベル文字列そのものを値にする
                values=scenario_labels,    # ← values もラベルのリスト
                on_value=set_selected_label,
            ),
            solara.Div(style={"height": "10px"}),
        ]
    )

    # 最終的に「コントロール」と「可視化」を縦に並べる
    return solara.Column(
        children=[
            controls,
            viz,  # ← viz() ではなく viz をそのまま渡す
        ]
    )
