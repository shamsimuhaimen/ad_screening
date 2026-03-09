# Terraform Setup

Infra in this directory provisions:
- `vscode-cpu` Compute Engine VM (`e2-standard-8`)
- `perma-disk` 200GB persistent SSD mounted at `/mnt/perma`
- Docker Engine + Docker Compose plugin installed via VM startup script

Boot disk is 30GB `pd-standard` (can align with free-tier standard disk allowance).

## Prerequisites
- Terraform >= 1.3
- `gcloud` authenticated to the correct account/project
- APIs enabled in `ad-screening-489618`:
  - `compute.googleapis.com`

## Apply
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

After the instance boots or is updated, Docker is installed and started. If you want to run `docker` without `sudo`, add your Linux user to the `docker` group separately.

## Connect
Use OS Login:
```bash
gcloud compute ssh vscode-cpu --zone northamerica-northeast2-a --project ad-screening-489618
```

## Destroy
`perma-disk` has `prevent_destroy = true`, so remove it intentionally only when needed.

To destroy VM only:
```bash
cd terraform
terraform destroy -target=google_compute_instance.vscode_cpu
```

To destroy disk too, first remove/override `prevent_destroy`, then destroy.
