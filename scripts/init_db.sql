-- Quant Platform Database Initialization
-- Run: mysql -u root -p < scripts/init_db.sql

CREATE DATABASE IF NOT EXISTS quant_platform
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;

USE quant_platform;

CREATE TABLE IF NOT EXISTS strategy (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(100) NOT NULL,
    type ENUM('stock_factor', 'etf_trend', 'etf_pairs', 'rotation', 'etf_reversal') NOT NULL,
    status ENUM('draft', 'backtesting', 'live', 'stopped') DEFAULT 'draft',
    config JSON,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_status (status),
    INDEX idx_type (type)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS backtest_result (
    id INT PRIMARY KEY AUTO_INCREMENT,
    strategy_id INT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    benchmark VARCHAR(20) DEFAULT 'SH000300',
    init_cash DECIMAL(15,2) DEFAULT 1000000,
    annual_return DECIMAL(10,4),
    sharpe_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(10,4),
    turnover DECIMAL(10,4),
    total_return DECIMAL(10,4),
    total_days INT,
    report_path VARCHAR(500),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id) ON DELETE CASCADE,
    INDEX idx_strategy (strategy_id),
    INDEX idx_dates (start_date, end_date)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS factor (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(200) NOT NULL,
    category ENUM('momentum', 'reversal', 'volatility', 'volume', 'fundamental', 'sentiment') NOT NULL,
    expression TEXT NOT NULL,
    description TEXT,
    ic_mean DECIMAL(10,4),
    ic_ir DECIMAL(10,4),
    rank_ic DECIMAL(10,4),
    long_short_return DECIMAL(10,4),
    source ENUM('manual', 'rd_agent') DEFAULT 'manual',
    status ENUM('draft', 'validated', 'active', 'deprecated') DEFAULT 'draft',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    INDEX idx_status (status),
    INDEX idx_source (source)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS position (
    id INT PRIMARY KEY AUTO_INCREMENT,
    strategy_id INT NOT NULL,
    instrument_id VARCHAR(20) NOT NULL,
    instrument_name VARCHAR(50),
    weight DECIMAL(10,6),
    shares INT,
    entry_price DECIMAL(10,4),
    entry_date DATE,
    current_price DECIMAL(10,4),
    pnl DECIMAL(15,2),
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id) ON DELETE CASCADE,
    INDEX idx_strategy (strategy_id),
    INDEX idx_instrument (instrument_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS trade_record (
    id INT PRIMARY KEY AUTO_INCREMENT,
    strategy_id INT NOT NULL,
    instrument_id VARCHAR(20) NOT NULL,
    side ENUM('buy', 'sell') NOT NULL,
    price DECIMAL(10,4) NOT NULL,
    volume INT NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    commission DECIMAL(10,2),
    trade_date DATETIME NOT NULL,
    status ENUM('pending', 'filled', 'partial', 'cancelled') DEFAULT 'pending',
    order_id VARCHAR(100),
    FOREIGN KEY (strategy_id) REFERENCES strategy(id) ON DELETE CASCADE,
    INDEX idx_strategy (strategy_id),
    INDEX idx_trade_date (trade_date),
    INDEX idx_instrument (instrument_id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS rd_agent_log (
    id INT PRIMARY KEY AUTO_INCREMENT,
    scenario VARCHAR(50) NOT NULL,
    iteration INT NOT NULL,
    factor_name VARCHAR(200),
    factor_expression TEXT,
    ic DECIMAL(10,4),
    icir DECIMAL(10,4),
    status ENUM('proposed', 'tested', 'validated', 'rejected') DEFAULT 'proposed',
    raw_output TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_scenario (scenario),
    INDEX idx_iteration (iteration),
    INDEX idx_status (status)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS data_update_log (
    id INT PRIMARY KEY AUTO_INCREMENT,
    data_type ENUM('stock_daily', 'etf_daily', 'fundamental', 'index_constituent') NOT NULL,
    trade_date DATE NOT NULL,
    instrument_count INT,
    record_count INT,
    status ENUM('success', 'partial', 'failed') NOT NULL,
    error_msg TEXT,
    duration_sec INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_type_date (data_type, trade_date),
    INDEX idx_status (status)
) ENGINE=InnoDB;

INSERT INTO strategy (name, type, status, description) VALUES
('ETF均线趋势', 'etf_trend', 'draft', '5日/20日均线交叉趋势跟踪策略'),
('ETF布林带回归', 'etf_reversal', 'draft', '20日布林带均值回归策略'),
('A股动量因子', 'stock_factor', 'draft', '20日动量因子选股策略'),
('A股低波动因子', 'stock_factor', 'draft', '20日低波动因子选股策略');

INSERT INTO factor (name, category, expression, description, source, status) VALUES
('MOM_20', 'momentum', 'close.pct_change(20)', '20日动量', 'manual', 'validated'),
('REV_5', 'reversal', '-close.pct_change(5)', '5日反转', 'manual', 'validated'),
('VOL_20', 'volatility', 'close.pct_change().rolling(20).std()', '20日波动率', 'manual', 'validated'),
('TURN_5', 'volume', 'turn.rolling(5).mean()', '5日平均换手率', 'manual', 'validated'),
('VR_20', 'volume', 'volume / volume.rolling(20).mean() - 1', '20日量比', 'manual', 'draft'),
('AMP', 'volatility', '(high - low) / close', '日内振幅', 'manual', 'draft');
