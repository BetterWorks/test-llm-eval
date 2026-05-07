#!/bin/bash
# Test all 6 features with a small number of test cases
# Usage: ./test_all_features.sh [max_cases]

MAX_CASES=${1:-5}

echo "========================================"
echo "Testing All Features"
echo "Max cases per feature: $MAX_CASES"
echo "========================================"
echo ""

FEATURES=(
  "writing_assistant"
  "goal_assist"
  "feedback_summary"
  "performance_summary"
  "meetings_summary"
  "skills_discovery"
)

SUCCESS_COUNT=0
FAIL_COUNT=0
FAILED_FEATURES=()

for feature in "${FEATURES[@]}"; do
  echo "========================================"
  echo "Testing: $feature"
  echo "========================================"
  
  if python cli.py run --feature "$feature" --max-cases "$MAX_CASES"; then
    echo "✅ $feature - SUCCESS"
    ((SUCCESS_COUNT++))
  else
    echo "❌ $feature - FAILED"
    ((FAIL_COUNT++))
    FAILED_FEATURES+=("$feature")
  fi
  
  echo ""
  sleep 2
done

echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total features: ${#FEATURES[@]}"
echo "Successful: $SUCCESS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -gt 0 ]; then
  echo ""
  echo "Failed features:"
  for failed in "${FAILED_FEATURES[@]}"; do
    echo "  - $failed"
  done
  exit 1
else
  echo ""
  echo "🎉 All features passed!"
  exit 0
fi
