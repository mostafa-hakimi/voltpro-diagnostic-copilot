import json
import time
import uuid

from langchain_core.messages import HumanMessage

from core.graph import support_app


def run_evaluation():
    print("[EVAL] Running automated benchmark against golden dataset...\n" + "=" * 65)

    with open("eval/golden_dataset.json", "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    results = []
    total_latency = 0.0
    passed_count = 0

    for item in golden_set:
        query = item["question"]
        # Refusal-type questions accept several valid phrasings (a real model
        # tested here correctly declined two out-of-scope questions without
        # using the exact word "documented" — e.g. "I am only authorized to
        # provide support for the VoltPro-X9000" is a correct refusal, not a
        # failure, so a single hardcoded keyword was the wrong check).
        expected_list = item.get("expected_keywords") or [item.get("expected_keyword", "")]
        category = item["category"]

        config = {
            "configurable": {"thread_id": f"eval_{uuid.uuid4()}"},
            "recursion_limit": 25,
        }

        start_time = time.time()
        response = support_app.invoke({"messages": [HumanMessage(content=query)]}, config=config)
        latency = round(time.time() - start_time, 2)
        total_latency += latency

        answer = response["messages"][-1].content
        is_passed = any(kw.lower() in answer.lower() for kw in expected_list)
        if is_passed:
            passed_count += 1

        results.append({
            "ID": item["id"],
            "Category": category,
            "Type": item.get("type", "answerable"),
            "Latency": f"{latency}s",
            "Status": "PASSED" if is_passed else "FAILED",
        })
        print(f"[{item['id']}] {category:<24} -> {'PASSED' if is_passed else 'FAILED'} ({latency}s)")

    accuracy = (passed_count / len(golden_set)) * 100
    avg_latency = round(total_latency / len(golden_set), 2)
    refusal_items = [r for r, g in zip(results, golden_set) if g.get("type") == "unanswerable"]
    refusal_passed = sum(1 for r in refusal_items if r["Status"] == "PASSED")

    print("\n" + "=" * 65)
    print("BENCHMARK SUMMARY:")
    print(f"Overall Accuracy       : {accuracy:.1f}% ({passed_count}/{len(golden_set)})")
    print(f"Average Latency        : {avg_latency}s")
    print(f"Correct Refusals       : {refusal_passed}/{len(refusal_items)} "
          f"(unanswerable questions the model correctly declined to guess on)\n")

    print("Markdown table for README.md:")
    print("| Query ID | Category | Target | Latency | Status |")
    print("|:---|:---|:---|:---:|:---:|")
    for r, item in zip(results, golden_set):
        status_icon = "PASS" if r["Status"] == "PASSED" else "FAIL"
        target = item.get("expected_keyword") or " / ".join(item.get("expected_keywords", []))
        print(f"| {r['ID']} | {r['Category']} | `{target}` | {r['Latency']} | {status_icon} |")


if __name__ == "__main__":
    run_evaluation()
