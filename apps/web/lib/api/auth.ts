import { get, post, put } from "@/lib/api/client";

export interface User {
  id: string;
  name: string;
  email: string;
  avatarUrl?: string;
}

export interface Session {
  user: User;
  accessToken: string;
  expiresAt: string;
}

export interface LoginInput {
  email: string;
  password: string;
}

export interface RegisterInput {
  name: string;
  email: string;
  password: string;
}

export interface AuthApi {
  me(): Promise<User>;
  login(input: LoginInput): Promise<Session>;
  register(input: RegisterInput): Promise<Session>;
  forgotPassword(email: string): Promise<void>;
  logout(): Promise<void>;
}

export const authApi: AuthApi = {
  me: () => get<User>("/v1/me"),
  login: (input) => post<Session>("/v1/auth/login", input),
  register: (input) => post<Session>("/v1/auth/register", input),
  forgotPassword: (email) => post<void>("/v1/auth/forgot-password", { email }),
  logout: () => put<void>("/v1/auth/logout"),
};
