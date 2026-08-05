#!/bin/bash
# gcc wrapper for compile_commands.json generation
# Intercepts gcc/cc compilation calls and logs them to /tmp/cc_log.jsonl

CMD=("$@")
INPUT=""
OUTPUT=""
IS_COMPILE=0

for arg in "${CMD[@]}"; do
    case "$arg" in
        -c) IS_COMPILE=1 ;;
        -o) continue ;;
        *.c) INPUT="$arg" ;;
        *.o) OUTPUT="$arg" ;;
    esac
done

# Call real gcc
/usr/bin/gcc "${CMD[@]}"
exit_code=$?

# If this was a compile, log it
if [ "$IS_COMPILE" -eq 1 ] && [ -n "$INPUT" ] && [ -n "$OUTPUT" ]; then
    CMD_STR=$(printf '%s\n' "${CMD[*]}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().strip()))" 2>/dev/null || echo '"gcc"')
    echo "{\"directory\":\"/home/wubu/wubuwizard\",\"command\":${CMD_STR},\"file\":\"$INPUT\",\"output\":\"$OUTPUT\"}" >> /tmp/cc_log.jsonl
fi

exit $exit_code