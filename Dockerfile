FROM registry.fiblab.net/general/dev:latest as builder

WORKDIR /build
COPY . /build
ENV PIP_NO_CACHE_DIR=1
RUN pip3 install --upgrade pip \ 
    && pip3 install pdoc \
    && ./scripts/gen_docs.sh

FROM node:20-alpine
ENV NODE_ENV=production
WORKDIR /home/node/app

# Install serve
RUN yarn global add serve

# Copy build files
RUN mkdir -p /home/node/app/build/docs
COPY --from=builder /build/docs ./build

EXPOSE 80

CMD serve build -p 80
