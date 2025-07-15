-- create database ai_tcbs;

CREATE TABLE public.chat_history (
	id serial4 NOT NULL,
	message_id varchar(255) NOT NULL,
	thread_id varchar(255) NOT NULL,
	"role" varchar(20) NOT NULL,
	"content" text NOT NULL,
	analysis_result json NULL,
	created_at timestamp NULL DEFAULT CURRENT_TIMESTAMP,
	updated_at timestamp NULL DEFAULT CURRENT_TIMESTAMP,
	CONSTRAINT chat_history_message_id_key UNIQUE (message_id),
	CONSTRAINT chat_history_pkey PRIMARY KEY (id)
);
CREATE INDEX idx_chat_history_created_at ON public.chat_history USING btree (created_at);
CREATE INDEX idx_chat_history_message_id ON public.chat_history USING btree (message_id);
CREATE INDEX idx_chat_history_thread_id ON public.chat_history USING btree (thread_id);

CREATE TABLE public.project_threads (
	id serial4 NOT NULL,
	thread_id varchar(255) NOT NULL,
	"name" varchar(255) NOT NULL,
	description text NULL,
	project_id varchar(255) NOT NULL,
	branch varchar(100) NULL,
	context_summary text NULL,
	is_active bool NULL,
	message_count int4 NULL,
	last_activity timestamp NULL,
	created_at timestamp NULL,
	updated_at timestamp NULL,
	api_method text NULL,
	documents text NULL,
	jira_links text NULL,
	api_path text NULL,
	api_documents text NULL,
	"references" text NULL,
	CONSTRAINT project_threads_pkey PRIMARY KEY (id)
);
CREATE INDEX ix_project_threads_id ON public.project_threads USING btree (id);
CREATE UNIQUE INDEX ix_project_threads_thread_id ON public.project_threads USING btree (thread_id);

ALTER TABLE public.chat_history ADD CONSTRAINT chat_history_thread_id_fkey FOREIGN KEY (thread_id) REFERENCES public.project_threads(thread_id) ON DELETE CASCADE;

CREATE TABLE public.projects (
	id serial4 NOT NULL,
	project_id varchar(255) NOT NULL,
	"name" varchar(255) NOT NULL,
	description text NULL,
	bitbucket_url varchar(500) NOT NULL,
	workspace varchar(255) NOT NULL,
	repository varchar(255) NOT NULL,
	default_branch varchar(100) NULL,
	commit_hash varchar(100) NULL,
	indexed_files int4 NULL,
	extracted_files int4 NULL,
	dependency_graph json NULL,
	status varchar(50) NULL,
	is_active bool NULL,
	created_at timestamp NULL,
	updated_at timestamp NULL,
	last_indexed_at timestamp NULL,
	CONSTRAINT projects_pkey PRIMARY KEY (id)
);
CREATE INDEX ix_projects_id ON public.projects USING btree (id);
CREATE UNIQUE INDEX ix_projects_project_id ON public.projects USING btree (project_id);

ALTER TABLE public.project_threads ADD CONSTRAINT project_threads_project_id_fkey FOREIGN KEY (project_id) REFERENCES public.projects(project_id);
