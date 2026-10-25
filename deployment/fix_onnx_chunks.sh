#!/bin/bash
set -u
OUT=/mnt/workspace/HoloMotion/models/holomotion-v2/exported
CH=$OUT/chunks2
URLS=/mnt/workspace/logs/onnx_fresh_urls.txt
TOTAL=1647887681
CHUNK=102992981
EXPECT_SHA=b0681976135eed572d1d20aefc52ec65c149bdc636f2594a6436fa7136a3fc86

cd "$CH" || { echo "NO_CHUNK_DIR"; exit 1; }
[ -s "$URLS" ] || { echo "NO_URLS"; exit 1; }

get_chunk() {
  i=$1
  base=$2
  s=$(( 10#$i * CHUNK ))
  e=$(( s + CHUNK - 1 ))
  [ "$e" -ge "$TOTAL" ] && e=$(( TOTAL - 1 ))
  exp=$(( e - s + 1 ))
  case "$i" in
    06) want=16413cd8cca99fa5bbfd48011ef191b0 ;;
    14) want=99224831341e3585b497f40c14238627 ;;
    *) echo "bad chunk $i"; return 1 ;;
  esac
  for k in 0 1 2 3; do
    line=$(( base + k ))
    url=$(sed -n "${line}p" "$URLS")
    [ -z "$url" ] && continue
    tmp="chunk_${i}.tmp"
    rm -f "$tmp"
    echo "[$(date +%H:%M:%S)] chunk_$i try$((k+1)) range=$s-$e"
    curl -sS -L --max-time 1800 -r "$s-$e" -o "$tmp" "$url"
    sz=$(stat -c%s "$tmp" 2>/dev/null || echo 0)
    got=$(md5sum "$tmp" 2>/dev/null | awk '{print $1}')
    if [ "$sz" -eq "$exp" ] && [ "$got" = "$want" ]; then
      mv "$tmp" "chunk_$i"
      echo "[$(date +%H:%M:%S)] chunk_$i OK size=$sz md5=$got"
      return 0
    fi
    echo "[$(date +%H:%M:%S)] chunk_$i BAD size=$sz md5=$got"
    rm -f "$tmp"
  done
  return 1
}

get_chunk 06 2 & p1=$!
get_chunk 14 6 & p2=$!
wait $p1 || f1=1
wait $p2 || f2=1
if [ "${f1:-0}" -ne 0 ] || [ "${f2:-0}" -ne 0 ]; then
  echo "MISSING_FAILED"
  exit 1
fi

echo '--- verify all chunks ---'
md5sum -c --quiet <<'EOF_MD5'
cbc156e3a3a84ec26b6d66423d57fc74  chunk_00
c569cffbb59ae07b74fb16471c0c2484  chunk_01
001ad8a71e6dd78edb59dee5afb70af5  chunk_02
32aee57d8625a2df451c878d23062cc9  chunk_03
4cf43ebeb2db1fb87a6788b3ea26891f  chunk_04
b4b7ac42c2cec47e6da05db58960f625  chunk_05
16413cd8cca99fa5bbfd48011ef191b0  chunk_06
11fe8ffd01f9cfa8fb9987a129849d3b  chunk_07
5813c5a1e7be6249f2f7f1b3a3caf1c6  chunk_08
9beaaac5bc97dade489e3452c9f3d140  chunk_09
a05424cab921433558d53a52d03505bf  chunk_10
b6d0504b50ebe97468f17ed74d56ec69  chunk_11
5078a8bc1fa789f4d98534a6d14dadd5  chunk_12
c0d8e09f05f476ec3e06505ba076da92  chunk_13
99224831341e3585b497f40c14238627  chunk_14
8c6ad4c2bdbfd7e0a37d31408d281867  chunk_15
EOF_MD5
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "CHUNK_MD5_FAILED"
  exit 1
fi

echo '--- assemble ---'
cat chunk_00 chunk_01 chunk_02 chunk_03 chunk_04 chunk_05 chunk_06 chunk_07 \
    chunk_08 chunk_09 chunk_10 chunk_11 chunk_12 chunk_13 chunk_14 chunk_15 \
    > "$OUT/model_10500.onnx"
cd "$OUT" || exit 1
sha=$(sha256sum model_10500.onnx | awk '{print $1}')
sz=$(stat -c%s model_10500.onnx)
echo "sha256=$sha size=$sz"
if [ "$sha" = "$EXPECT_SHA" ] && [ "$sz" -eq "$TOTAL" ]; then
  rm -rf chunks2
  echo "ONNX_FIXED $(date)" | tee /mnt/workspace/logs/onnx_fix_receipt.txt
else
  echo "SHA_MISMATCH"
  exit 1
fi
