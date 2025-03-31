aws configure list

# Echo
TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo $TIME : Start!

AWS_SOURCE=s3://radiantearth/cloud-cover-detection-challenge/final/public
aws s3 cp $AWS_SOURCE ./public --recursive --endpoint-url=https://data.source.coop &> ./log_download.txt

# Echo
TIME="`date "+%Y-%m-%d %H:%M:%S"`"
echo $TIME : Done !



# scp -i ~/.ssh/s2-cloudcover.pem ~/.aws/config ubuntu@44.223.27.233:~/.aws