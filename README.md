# SMS-Spam-Detection-with-Spark
In this project we implement a classification task to detect spam in SMS showing show how distributed computation may improve efficiency when dealing with Machine Learning techniques.

## 1. Terraform
Terraform is an open-source infrastructure as code software tool created by HashiCorp. Users define and provide data center infrastructure using a declarative configuration language. We used it to write all configuration necessary to make cluster on AWS in few time.\
Terraform must therefore be installed [Link to Terraform installation](https://www.terraform.io/docs/cli/install/apt.html).
## 2. AWS
Amazon Web Services (AWS) is a subsidiary of Amazon providing on-demand cloud computing platforms and APIs.\
After login to AWS:
- go to IAM section > `Users` > `Add User`, write a name, check Programmatic Access and go next. Link an existent policy `AmazonEC2FullAccess` and go on till the end when show button with `create new user`, after this click you must copy **Access Key Id** and **Secret Access Key** (this one will no longer be shown).
- create a key pairs named amzkey in PEM file format. Follow the guide on [AWS DOCS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair). Download the key and put it in the `<project>` folder.
## 3. Download project and edit it
Download this project > go to the root of project > create new file `terraform.tfvars` and write:
```
access_key = "<Access Key Id>"
secret_key = "<Secret Access Key>"
token      = ""
```
Open the file `variables.tf` and pay attention to change value of:
- "region", must be the same of key-pair create above;
- "access_key", "secret_key" are the same above;
- "datanode_count", "instance_type" choose number and type of the EC2 for your project (for this project we have tested 5 t2.small)
- "subnet_id" go to aws and search [Subnets](https://console.aws.amazon.com/vpc/home?region=us-east-1#subnets:), copy the name of `Subnet ID` with *IPv4 CIDR = 172.31.0.0/20*

Open a terminal and generate a new ssh-key
```
ssh-keygen -f <PROJECT_PATH>/localkey
```
## 4. Start Terraform and make aws instances
From path of project run these commands:
```
terraform init
terraform apply
```
At the end the terminal prints some `PUBLIC DNS`, these are the instances. You can access them in ssh with this command:
```
ssh -i <PROJECT_PATH>/amzkey.pem ubuntu@<PUBLIC DNS>
```
## 5. Start Hadoop and Spark
Access to master and execute this commands to start Hadoop, master and slaves:
```
sh hadoop-start-master.sh
$SPARK_HOME/sbin/start-master.sh
hdfs dfs -put ~/dataset.csv /dataset.csv
$SPARK_HOME/sbin/start-slaves.sh
```
## 6. Start application
The time is now! Run the following command:
```
python3 main.py
```
or if you want to set some parameters of Spark:
```
$SPARK_HOME/bin/spark-submit --master spark://s01:7077 --executor-cores 2 --executor-memory 2g main.py
```
## 6. Shut down all
At the end you can remove from AWS everything you have created with this command on your machine in the folder project:
```
terraform destroy
```
