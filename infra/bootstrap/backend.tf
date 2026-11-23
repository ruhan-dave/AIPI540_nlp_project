# The state bucket's own backend is DISABLED while the bucket does not exist:
# the bootstrap root necessarily starts with local state.
#
#   1. terraform init && terraform apply                     # local state
#   2. cp backend.hcl.example backend.hcl, then fill it in from the
#      `backend_config` output
#   3. uncomment the block below
#   4. terraform init -migrate-state -backend-config=backend.hcl
#
# To tear the bucket down (the final, separately reviewed teardown step),
# comment the block out again and migrate the state back to local state.
#
# terraform {
#   backend "s3" {
#     use_lockfile = true
#     encrypt       = true
#   }
# }
