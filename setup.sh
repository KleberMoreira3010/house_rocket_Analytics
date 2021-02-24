mkdir -p ~/.streamlit/

echo "\
[general]\n\
email=\"klb_moreira@hotmail.com\"\n\
" > ~/.streamlit/credentials.toml


echo "\
[server]\n\
headless=true\n\
enableCORS=flase\n\
port=$PORT\n\
" >~/.streamlit/config.toml