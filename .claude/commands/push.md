Follow these steps in order to push the project. Complete each step fully before moving to the next:

1. Check the working tree and confirm the intended file set — verify that no untracked files needed by the project are missing from version control
2. Review the diff so the pushed scope is explicit
3. Run the smallest useful validation for the changed code — when tagging a release, run the broadest available check rather than the smallest
4. Update the dependency snapshot (e.g., `pip freeze > requirements.txt` or `conda env export > environment.yml`) so the tagged version locks the exact packages used
5. Choose the next version number using the repo's versioning scheme
6. Add or update the changelog entry for that version
7. Update any stale documentation or guidance affected by the change
8. Stage only the files intended for the push
9. Create a commit with a versioned, descriptive message
10. Push the branch and, when the repo uses release tags, create and push the matching tag
