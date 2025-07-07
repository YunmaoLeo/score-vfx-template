#!/bin/bash
rm -rf release
mkdir -p release

cp -rf QVKRT *.{hpp,cpp,txt,json} LICENSE release/

mv release score-addon-qvkrt
7z a score-addon-qvkrt.zip score-addon-qvkrt
