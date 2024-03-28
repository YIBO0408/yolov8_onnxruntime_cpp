rm -r ../build

mkdir ../build

cd ../build

cmake ..

make -j 16

cd ../scripts

sh run.sh