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
      bgColor: "bg-blue-50 dark:bg-slate-800",
      iconBg: "bg-blue-600",
      iconColor: "text-blue-600 dark:text-blue-500",
      borderColor: "border-blue-200 dark:border-slate-700"
    },
    {
      icon: TrendingUp,
      label: "Accuracy Rate",
      value: "99.2%",
      bgColor: "bg-green-50 dark:bg-slate-800",
      iconBg: "bg-green-600",
      iconColor: "text-green-600 dark:text-green-500",
      borderColor: "border-green-200 dark:border-slate-700"
    },
    {
      icon: Zap,
      label: "Real-time Updates",
      value: "24/7",
      bgColor: "bg-orange-50 dark:bg-slate-800",
      iconBg: "bg-orange-600",
      iconColor: "text-orange-600 dark:text-orange-500",
      borderColor: "border-orange-200 dark:border-slate-700"
    },
    {
      icon: CheckCircle,
      label: "Fact-Checked",
      value: "100%",
      bgColor: "bg-cyan-50 dark:bg-slate-800",
      iconBg: "bg-cyan-600",
      iconColor: "text-cyan-600 dark:text-cyan-500",
      borderColor: "border-cyan-200 dark:border-slate-700"
    },
  ];

  return (
    <section className="py-12 bg-white dark:bg-slate-800 border-y border-slate-200 dark:border-slate-700">
      <div className="max-w-7xl mx-auto px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <div
                key={index}
                className={`${stat.bgColor} rounded-lg p-6 border ${stat.borderColor} hover:shadow-lg transition-shadow`}
              >
                <div className="flex items-center gap-4">
                  <div className={`p-3 rounded-lg ${stat.iconBg}`}>
                    <Icon className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <div className={`text-2xl font-bold ${stat.iconColor}`}>
                      {stat.value}
                    </div>
                    <div className="text-xs font-medium text-slate-600 dark:text-slate-400 mt-0.5">
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

