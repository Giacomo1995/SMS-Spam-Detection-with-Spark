# SMS Spam Detection with Spark
Implementation of a classification task to detect spam in text messages using Spark.

## 1. Terraform
Terraform is an open-source infrastructure as a code software tool created by HashiCorp. Users define and provide data center infrastructure using a declarative configuration language.

[Link to install Terraform](https://www.terraform.io/docs/cli/install/apt.html).
## 2. AWS
Amazon Web Services (AWS) is a subsidiary of Amazon providing on-demand cloud computing platforms and APIs.

After login to AWS:
- Go to IAM section > `Users` > `Add User`, write a name, check Programmatic Access and proceed. Link an existent policy `AmazonEC2FullAccess` and go on till the `create new user` button, click on it, and save **Access Key Id** and **Secret Access Key** (it will no longer be shown).
- Create a key pair named amzkey in a PEM file format. Follow the guide on [AWS DOCS](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html#having-ec2-create-your-key-pair). Download the key and move it in the `<project>` folder.
## 3. Download project and edit it
Download this project > go to the root of the project > create a new file `terraform.tfvars` and write:
```
access_key = "<Access Key Id>"
secret_key = "<Secret Access Key>"
token      = ""
```
Open the file `variables.tf` and pay attention to change values of:
- "region": must be the same of key-pair created above;
- "access_key" and "secret_key": are the same listed above;
- "datanode_count" and "instance_type": choose number and type of the EC2 for your project (for this project, we have tested six t2.medium instances)
- "subnet_id": go to AWS and search [Subnets](https://console.aws.amazon.com/vpc/home?region=us-east-1#subnets:), copy the name of `Subnet ID` with *IPv4 CIDR = 172.31.0.0/20*

Open the file `main.tf` and in row 22 substitute `*.*.*.*` with your public IP.

Open the terminal and generate a new ssh-key
```
ssh-keygen -f <PROJECT_PATH>/localkey
```
## 4. Start Terraform and make AWS instances
From the root of the project run these commands:
```
terraform init
terraform apply
```
At the end the terminal will print several `PUBLIC DNS`: these are the instances. You can access them in ssh with this command:
```
ssh -i <PROJECT_PATH>/amzkey.pem ubuntu@<PUBLIC DNS>
```
## 5. Start Hadoop and Spark
Access the master and run the following commands to start Hadoop, master, and slaves:
```
sh hadoop-start-master.sh
$SPARK_HOME/sbin/start-master.sh
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
## 7. Shut down
In the end, you can remove from AWS everything you have created with this command on your machine in the folder project:
```
terraform destroy
```
## See also
- [Hadoop/Spark with Terraform on AWS](https://github.com/conema/spark-terraform.git)
- [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
