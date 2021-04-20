#!/bin/bash
aws configure set default.region us-east-1
export AWS_ACCESS_KEY_ID=AKIA3X3UYX5IV3L3RGYV
export AWS_SECRET_ACCESS_KEY=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
aws ssm start-session --target i-0400c09aebb44ad61 --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["9443"], "localPortNumber":["9443"]}'
                                 Team Name : uneversleepwell-unsw

