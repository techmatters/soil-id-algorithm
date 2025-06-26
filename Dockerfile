# Stage 1: Build database with preloaded data
FROM postgis/postgis:16-3.5 as builder

ARG POSTGRES_DB=soil_id
ARG POSTGRES_USER=postgres
ARG POSTGRES_PASSWORD=postgres
ARG DATABASE_DUMP_FILE=Data/soil_id_db.dump

ENV POSTGRES_DB=${POSTGRES_DB}
ENV POSTGRES_USER=${POSTGRES_USER}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

# Copy dump into builder stage
COPY ${DATABASE_DUMP_FILE} /tmp/soil_id_db.dump

# Initialize database and preload it
RUN mkdir -p /var/lib/postgresql/data \
 && chown -R ${POSTGRES_USER}:${POSTGRES_USER} /var/lib/postgresql

USER ${POSTGRES_USER}

# Initialize database system
RUN /usr/lib/postgresql/16/bin/initdb -D /var/lib/postgresql/data

# Start database in background, preload dump, shut it down
RUN pg_ctl -D /var/lib/postgresql/data -o "-c listen_addresses=''" -w start \
  && createdb -U ${POSTGRES_USER} ${POSTGRES_DB} \
  && psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "CREATE EXTENSION postgis;" \
  && pg_restore -U ${POSTGRES_USER} -d ${POSTGRES_DB} /tmp/soil_id_db.dump \
  && psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "CLUSTER hwsd2_segment USING hwsd2_segment_shape_idx;" \
  && psql -U ${POSTGRES_USER} -d ${POSTGRES_DB} -c "ANALYZE;" \
  && pg_ctl -D /var/lib/postgresql/data -m fast -w stop

# Stage 2: Final image with loaded data
FROM postgis/postgis:16-3.5

COPY --from=builder /var/lib/postgresql/data /var/lib/postgresql/data

# Create a Docker-friendly pg_hba.conf
USER root
RUN echo "local   all             all                                     trust" > /var/lib/postgresql/data/pg_hba.conf \
 && echo "host    all             all             127.0.0.1/32            trust" >> /var/lib/postgresql/data/pg_hba.conf \
 && echo "host    all             all             ::1/128                 trust" >> /var/lib/postgresql/data/pg_hba.conf \
 && echo "host    all             all             0.0.0.0/0               trust" >> /var/lib/postgresql/data/pg_hba.conf \
 && chown ${POSTGRES_USER}:${POSTGRES_USER} /var/lib/postgresql/data/pg_hba.conf

ENV POSTGRES_DB=${POSTGRES_DB}
ENV POSTGRES_USER=${POSTGRES_USER}
ENV POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

USER ${POSTGRES_USER}
