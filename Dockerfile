# Stage 1: Build database with preloaded data
FROM postgis/postgis:16-3.5 as builder

ENV POSTGRES_DB=soil_id
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Copy dump into builder stage
COPY Data/soil_id_db.dump /tmp/soil_id_db.dump

# Initialize database and preload it
RUN mkdir -p /var/lib/postgresql/data \
 && chown -R postgres:postgres /var/lib/postgresql

USER postgres

# Initialize database system
RUN /usr/lib/postgresql/16/bin/initdb -D /var/lib/postgresql/data

# Start database in background, preload dump, shut it down
RUN pg_ctl -D /var/lib/postgresql/data -o "-c listen_addresses=''" -w start \
 && createdb -U postgres soil_id \
 && psql -U postgres -d $POSTGRES_DB -c "CREATE EXTENSION postgis;" \
 && pg_restore -U postgres -d soil_id /tmp/soil_id_db.dump \
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
 && chown postgres:postgres /var/lib/postgresql/data/pg_hba.conf

ENV POSTGRES_DB=soil_id
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

USER postgres
