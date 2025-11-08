"use client";

import { TrendingUp, Shield, Zap, CheckCircle } from "lucide-react";

interface StatsSectionProps {
  totalArticles: number;
}

export default function StatsSection({ totalArticles }: StatsSectionProps) {
  const stats = [
    {
      icon: Shield,
      label: "Articles Analyzed",
      value: totalArticles.toLocaleString(),
      iconBg: "bg-blue-600",
      iconColor: "text-blue-400"
    },
    {
      icon: TrendingUp,
      label: "Accuracy Rate",
      value: "99.2%",
      iconBg: "bg-green-500",
      iconColor: "text-green-400"
    },
    {
      icon: Zap,
      label: "Real-time Updates",
      value: "24/7",
      iconBg: "bg-yellow-500",
      iconColor: "text-yellow-400"
    },
    {
      icon: CheckCircle,
      label: "Fact-Checked",
      value: "100%",
      iconBg: "bg-cyan-500",
      iconColor: "text-cyan-400"
    },
  ];

  return (
    <section className="py-12 bg-zinc-900 border-y border-zinc-800">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div
                key={index}
                className="bg-zinc-800 rounded-lg p-6 border border-zinc-700 hover:border-zinc-600 hover:shadow-lg transition-all"
              >
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-lg ${stat.iconBg}`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <div className={`text-2xl font-bold ${stat.iconColor}`}>
                      {stat.value}
                    </div>
                    <div className="text-xs font-medium text-gray-400 mt-0.5">
                      {stat.label}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

