// Where the website finds its engine. scripts/site/build_site.py fills these in for deployment.
window.HD_CONFIG = {
  space: "__SPACE__", // the Hugging Face Space running app/api.py, e.g. "user/HumanDiT-engine"
  local: "http://localhost:7861", // app/api.py on this machine, used when the page itself is on localhost
  links: { github: "", model: "", space: "" },
};
