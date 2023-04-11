FROM python:3.10.6 as build
MAINTAINER zhangwenhui "zhangwenhui@sspedu.com"

ENV DOC_ENV=test
# set working directory
ADD docs/requirements.txt /data/www/
WORKDIR /data/www/
RUN python3 -m pip install -i https://mirrors.cloud.tencent.com/pypi/simple -r requirements.txt
ADD . /data/www/alphamed-docs/

WORKDIR /data/www/alphamed-docs/
RUN echo " ------------------Web打包开始--------------------"
RUN jupyter-book build docs/
# install app dependencies
RUN echo " ------------------Web打包完成 --------------------"


FROM nginx:1.19.2
RUN rm /etc/nginx/nginx.conf /etc/nginx/conf.d/default.conf
COPY nginx.prod.conf /etc/nginx/nginx.conf
COPY --from=build /data/www/alphamed-docs/docs/_build /usr/share/nginx
# ADD docs/_build /usr/share/nginx
EXPOSE 9000
CMD ["nginx", "-g", "daemon off;"]