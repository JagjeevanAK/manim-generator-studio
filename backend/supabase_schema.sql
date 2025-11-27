-- Create the manim_jobs table
create table public.manim_jobs (
  id uuid not null primary key,
  created_at timestamp with time zone default timezone('utc'::text, now()) not null,
  updated_at timestamp with time zone default timezone('utc'::text, now()) not null,
  status text not null,
  prompt text,
  code text,
  url text,
  message text
);

-- Enable Row Level Security (RLS)
alter table public.manim_jobs enable row level security;

-- Create a policy that allows anyone to read/write (for development simplicity)
-- WARNING: In production, you should restrict this to authenticated users
create policy "Enable all access for all users" on public.manim_jobs
for all using (true) with check (true);

-- Create a storage bucket for videos if it doesn't exist
insert into storage.buckets (id, name, public)
values ('manim-renders', 'manim-renders', true)
on conflict (id) do nothing;

-- Set up storage policy to allow public access
create policy "Public Access" on storage.objects for all using ( bucket_id = 'manim-renders' );
