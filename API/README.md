# LandPKS_API_Epic

gcloud compute instances create landpks-api-soilid --image=debian-8 --machine-type=g1-small  --scopes userinfo-email,cloud-platform --metadata-from-file startup-script=gce/startup-script.sh --zone us-west1-a --tags http-server

gcloud compute instances get-serial-port-output landpks-api-soilid --zone us-west1-a

gcloud compute firewall-rules create default-allow-http-8081 --allow tcp:8081 --source-ranges 0.0.0.0/0 --target-tags http-server --description "Allow port 8081 access to http-server"