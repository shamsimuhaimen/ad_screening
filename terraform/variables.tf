variable "project_id" {
  description = "GCP project where the CPU instance will be created."
  type        = string
  default     = "ad-screening-489618"
}

variable "region" {
  description = "Region that hosts the subnet and static IP."
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Zone where the compute instance will run."
  type        = string
  default     = "northamerica-northeast2-a"
}

variable "machine_type" {
  description = "Compute Engine machine type that provides at least 16GB memory."
  type        = string
  default     = "e2-standard-8"
}

variable "perma_disk_size_gb" {
  description = "Size of the additional persistent SSD disk."
  type        = number
  default     = 100
}
