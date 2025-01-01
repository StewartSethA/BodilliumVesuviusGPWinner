i=esuvius
for f in `ps -aux | grep python | grep $i | awk '{print $2}'`; do kill -9 $f; done
for f in `ps -aux | grep python | grep croll | awk '{print $2}'`; do kill -9 $f; done
for f in `ps -aux | grep python | grep 64x64 | awk '{print $2}'`; do kill -9 $f; done
for f in `ps -aux | grep python | grep 1x | awk '{print $2}'`; do kill -9 $f; done
for f in `ps -aux | grep "python]" | awk '{print $2}'`; do kill -9 $f; done
for f in `ps -aux | grep "<defunct>" | awk '{print $2}'`; do kill -9 $f; done
