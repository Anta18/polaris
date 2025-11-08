import mongoose from "mongoose";

const MONGO_URI = process.env.MONGO_URI || "mongodb://127.0.0.1:27017/polaris_news_db";
const AUTH_SOURCE = process.env.MONGO_AUTH_SOURCE || "admin";

declare global {
  // eslint-disable-next-line no-var
  var __mongoose: { conn: typeof mongoose | null; promise: Promise<typeof mongoose> | null } | undefined;
}

if (!global.__mongoose) {
  global.__mongoose = { conn: null, promise: null };
}

export async function connectDB() {
  if (global.__mongoose!.conn) return global.__mongoose!.conn;

  if (!global.__mongoose!.promise) {
    global.__mongoose!.promise = mongoose
      .connect(MONGO_URI, { authSource: AUTH_SOURCE } as any)
      .then((m) => m);
  }
  console.log("Connecting to MongoDB...");
  console.log(MONGO_URI);

  global.__mongoose!.conn = await global.__mongoose!.promise;
  return global.__mongoose!.conn;
}
