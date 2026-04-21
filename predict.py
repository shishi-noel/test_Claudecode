"""
離職予測 CLI
Usage:
    python predict.py --age 35 --gender 0 --tenure 5 --night_shifts 8 --stress 7.5
"""
import argparse
from turnover_model import predict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="従業員の離職確率を予測します")
    p.add_argument("--age",          type=float, required=True,  help="年齢 (例: 35)")
    p.add_argument("--gender",       type=int,   required=True,  help="性別 0=男性 1=女性")
    p.add_argument("--tenure",       type=float, required=True,  help="勤務年数 (例: 5)")
    p.add_argument("--night_shifts", type=float, required=True,  help="月あたり夜勤回数 (例: 8)")
    p.add_argument("--stress",       type=float, required=True,  help="ストレス指標 1〜10 (例: 7.5)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (1 <= args.stress <= 10):
        raise ValueError("ストレス指標は 1〜10 の値を入力してください。")
    if args.gender not in (0, 1):
        raise ValueError("性別は 0（男性）または 1（女性）を入力してください。")

    prob = predict(
        age=args.age,
        gender=args.gender,
        tenure=args.tenure,
        night_shifts=args.night_shifts,
        stress=args.stress,
    )

    print("\n===== 離職予測結果 =====")
    print(f"  年齢        : {args.age:.0f} 歳")
    print(f"  性別        : {'女性' if args.gender == 1 else '男性'}")
    print(f"  勤務年数    : {args.tenure:.0f} 年")
    print(f"  夜勤回数    : {args.night_shifts:.0f} 回/月")
    print(f"  ストレス指標: {args.stress:.1f} / 10")
    print(f"------------------------")
    print(f"  離職確率    : {prob * 100:.1f} %")
    if prob >= 0.7:
        print("  リスク判定  : ⚠ 高リスク")
    elif prob >= 0.4:
        print("  リスク判定  : △ 中リスク")
    else:
        print("  リスク判定  : ○ 低リスク")
    print("========================\n")


if __name__ == "__main__":
    main()
