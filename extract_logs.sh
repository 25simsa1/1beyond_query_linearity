#!/bin/bash
# Emit step,train_loss,val_loss CSVs in Jiwon's schema for the runs in my write-up.
cd "$HOME/beyond_query_linearity" || exit 1
OUT=/tmp/wu_logs
rm -rf "$OUT"; mkdir -p "$OUT"

emit () {  # emit <outfile> <src.out> [more src.out ...]
  local dst="$OUT/$1"; shift
  echo "step,train_loss,val_loss" > "$dst"
  for f in "$@"; do
    [ -f "$f" ] || { echo "MISSING $f" >&2; continue; }
    grep -E '^step [0-9]+: train loss' "$f" \
      | sed -E 's/^step ([0-9]+): train loss ([0-9.]+), val loss ([0-9.]+).*/\1,\2,\3/'
  done | sort -t, -k1,1n -u >> "$dst"
  printf '%-52s %4d rows\n' "$1" "$(( $(wc -l < "$dst") - 1 ))"
}

# GPT-3 Large, linear query + QK-norm : the LR sweep
emit gpt3large_linear_qknorm_lr7p5e4.csv        qkn_lr7p5e4_9776.out
emit gpt3large_linear_qknorm_lr1p5e3_seed43.csv qkn_lr1p5e3_9777.out
emit gpt3large_linear_qknorm_lr1p5e3_seed44.csv qkn_lr1p5e3_s44_9924.out
emit gpt3large_linear_qknorm_lr2p55e3.csv       qkn_lr2p55e3_9778.out
emit gpt3large_linear_qknorm_lr2p5e4_anchor.csv orig_gpt3L_s43_qknorm_9645.out

# GPT-3 Large, residual-GELU
emit gpt3large_resgelu_knormph_s1_lr1p5e3.csv   resg_knorm_ph_s1_9944.out
emit gpt3large_resgelu_s0p5_seed43.csv          resg_gpt3L_s43_9436.out
emit gpt3large_resgelu_s0p5_seed44.csv          resg_gpt3L_s44_9647.out
emit gpt3large_resgelu_s0p5_seed45.csv          resg_gpt3L_s45_9780.out resg_s45_resume_10127.out

# GPT-3 Large, plain linear (no mitigation)
emit gpt3large_linear_plain_seed42.csv          gpt3L_orig_8952.out
emit gpt3large_linear_plain_seed43_diverged.csv orig_gpt3L_s43_9435.out
emit gpt3large_linear_plain_seed44_diverged.csv orig_gpt3L_s44_9779.out

# GPT-3 Large, plain linear under D2Z (linear decay to zero)
emit gpt3large_linear_d2z_lr1e3.csv             d2z_plain_lr1e3_10140.out
emit gpt3large_linear_d2z_lr2e3.csv             d2z_plain_lr2e3_10141.out

# GPT-3 small, D2Z learning-rate sweep at 60k
emit gpt3small_d2z_lr6e4_60k.csv                p3s_d2z_lr6e4_60k_10130.out
emit gpt3small_d2z_lr1p2e3_60k.csv              p3s_d2z_lr1p2e3_60k_10131.out
emit gpt3small_d2z_lr2p4e3_60k.csv              p3s_d2z_lr2p4e3_60k_10132.out
emit gpt3small_d2z_lr4p8e3_60k.csv              p3s_d2z_lr4p8e3_60k_10133.out
emit gpt3small_d2z_lr9p6e3_60k.csv              p3s_d2z_lr9p6e3_60k_10134.out

# GPT-3 small, the 180k arm
emit gpt3small_d2z_lr6e4_180k.csv               p3s_d2z_lr6e4_180k_10135.out

echo
echo "=== 10140 final state ==="
sacct -j 10140 --format=JobID%9,JobName%20,State%14,Elapsed,ExitCode -X 2>/dev/null
echo "=== attention-logit trace, last line of each D2Z run ==="
grep 'attn_logit' d2z_plain_lr1e3_10140.out | tail -1
grep 'attn_logit' d2z_plain_lr2e3_10141.out | tail -1
