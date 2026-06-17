/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DECADIC_HTTP?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
