terraform {
  required_version = ">= 1.3.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

locals {
  perma_disk_name       = "perma-disk"
  perma_disk_mount_path = "/mnt/perma"
}

resource "google_compute_instance" "vscode_cpu" {
  name         = "vscode-cpu"
  machine_type = var.machine_type
  zone         = var.zone

  metadata = {
    enable-oslogin = "TRUE"
  }

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2204-lts"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"

    access_config {
      # Empty block keeps an ephemeral external IP.
    }
  }

  attached_disk {
    source      = google_compute_disk.perma_disk.id
    device_name = local.perma_disk_name
    mode        = "READ_WRITE"
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = true
    preemptible         = false
  }

  metadata_startup_script = <<-EOT
    #!/bin/bash
    set -euo pipefail

    DISK_BY_ID="/dev/disk/by-id/google-${local.perma_disk_name}"
    MOUNT_POINT="${local.perma_disk_mount_path}"

    for attempt in $(seq 1 10); do
      if [[ -e "$${DISK_BY_ID}" ]]; then
        break
      fi
      sleep 5
    done

    if [[ ! -e "$${DISK_BY_ID}" ]]; then
      logger "perma-disk: $${DISK_BY_ID} not found; skipping mount"
      exit 0
    fi

    if ! blkid "$${DISK_BY_ID}" >/dev/null 2>&1; then
      mkfs.ext4 -F "$${DISK_BY_ID}"
    fi

    mkdir -p "$${MOUNT_POINT}"

    if ! grep -qs "$${DISK_BY_ID}" /etc/fstab; then
      echo "$${DISK_BY_ID} $${MOUNT_POINT} ext4 discard,defaults,nofail 0 2" >> /etc/fstab
    fi

    mountpoint -q "$${MOUNT_POINT}" || mount "$${MOUNT_POINT}"

    chmod 777 "$${MOUNT_POINT}"

    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y libgl1 libglib2.0-0 wget
  EOT
}

resource "google_compute_disk" "perma_disk" {
  name = local.perma_disk_name
  type = "pd-ssd"
  zone = var.zone
  size = var.perma_disk_size_gb

  lifecycle {
    prevent_destroy = true
    ignore_changes  = [snapshot]
  }
}
