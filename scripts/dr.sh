IMG=`grep -m 1 image .gitlab-ci.yml | awk '{print $2}'` 
echo $IMG
docker run  --user $(id -u):$(id -g) \
    --rm -t \
    -v `pwd`:/builds/app $IMG /bin/sh -c "cd /builds/app; $@"
