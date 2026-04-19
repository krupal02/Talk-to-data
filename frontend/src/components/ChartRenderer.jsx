/**
 * ChartRenderer – renders Recharts visualisations based on chart_type.
 *
 * Supports bar, grouped_bar, pie, and line charts with accessible colours,
 * descriptive tooltips, and labelled axes.
 */

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  PieChart, Pie, Cell,
  LineChart, Line,
  ResponsiveContainer,
} from 'recharts';

// Accessible colour palette (avoids red/green only)
const COLORS = [
  '#818cf8', // Indigo 400
  '#34d399', // Emerald 400
  '#fbbf24', // Amber 400
  '#f472b6', // Pink 400
  '#38bdf8', // Sky 400
  '#a78bfa', // Violet 400
  '#fb923c', // Orange 400
  '#2dd4bf', // Teal 400
  '#e879f9', // Fuchsia 400
  '#4ade80', // Green 400
];

const CUSTOM_TOOLTIP_STYLE = {
  backgroundColor: 'rgba(15, 23, 42, 0.95)',
  border: '1px solid rgba(99, 102, 241, 0.3)',
  borderRadius: '8px',
  padding: '10px 14px',
  boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)',
};

function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={CUSTOM_TOOLTIP_STYLE}>
      <p className="text-surface-100 font-semibold text-sm mb-1">{label}</p>
      {payload.map((entry, idx) => (
        <p key={idx} className="text-sm" style={{ color: entry.color || '#a5b4fc' }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toLocaleString() : entry.value}
        </p>
      ))}
    </div>
  );
}

function renderBarChart(data, title) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
        <XAxis
          dataKey="name"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        <Bar dataKey="value" name={title || 'Value'} radius={[6, 6, 0, 0]}>
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function renderGroupedBarChart(data, title) {
  // Detect all numeric keys in the data (excluding 'name' and 'delta')
  const numericKeys = data.length > 0
    ? Object.keys(data[0]).filter(
        (k) => k !== 'name' && k !== 'delta' && typeof data[0][k] === 'number'
      )
    : [];

  // Friendly labels for period comparison
  const labelMap = {
    period_1: 'Earlier Period',
    period_2: 'Later Period',
    mean: 'Average',
    min: 'Minimum',
    max: 'Maximum',
    median: 'Median',
  };

  return (
    <ResponsiveContainer width="100%" height={350}>
      <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
        <XAxis
          dataKey="name"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        {numericKeys.map((key, index) => (
          <Bar
            key={key}
            dataKey={key}
            name={labelMap[key] || key}
            fill={COLORS[index % COLORS.length]}
            radius={[4, 4, 0, 0]}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function renderPieChart(data, title) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={110}
          paddingAngle={3}
          dataKey="value"
          nameKey="name"
          label={({ name, percentage }) =>
            percentage ? `${name} (${percentage}%)` : name
          }
          labelLine={{ stroke: '#475569' }}
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ color: '#94a3b8', fontSize: 12 }}
          formatter={(value) => <span className="text-surface-200">{value}</span>}
        />
      </PieChart>
    </ResponsiveContainer>
  );
}

function renderLineChart(data, title) {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(51, 65, 85, 0.4)" />
        <XAxis
          dataKey="name"
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: '#94a3b8', fontSize: 12 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
          tickFormatter={(v) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        <Line
          type="monotone"
          dataKey="value"
          name={title || 'Value'}
          stroke="#818cf8"
          strokeWidth={2.5}
          dot={{ fill: '#818cf8', r: 4, strokeWidth: 2, stroke: '#0f172a' }}
          activeDot={{ r: 6, stroke: '#818cf8', strokeWidth: 2, fill: '#0f172a' }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function ChartRenderer({ chartType, chartData, title }) {
  if (!chartData || chartData.length === 0) {
    return (
      <div className="text-center py-10 text-surface-200/40 text-sm">
        No chart data available
      </div>
    );
  }

  const renderers = {
    bar: renderBarChart,
    grouped_bar: renderGroupedBarChart,
    pie: renderPieChart,
    line: renderLineChart,
  };

  const render = renderers[chartType] || renderBarChart;

  return (
    <div id="chart-container" className="chart-container">
      {title && (
        <h3 className="text-sm font-medium text-surface-200/60 mb-2 text-center">
          {title}
        </h3>
      )}
      {render(chartData, title)}
    </div>
  );
}
