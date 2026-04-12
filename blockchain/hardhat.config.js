require("@nomicfoundation/hardhat-toolbox");
require("dotenv").config();

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.20",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200,
      },
    },
  },

  networks: {
    // Local Hardhat node (npx hardhat node)
    localhost: {
      url: "http://127.0.0.1:8545",
    },

    // Sepolia testnet
    // Requires SEPOLIA_RPC_URL and PRIVATE_KEY in blockchain/.env
    sepolia: {
      url: process.env.SEPOLIA_RPC_URL || "",
      accounts:
        process.env.PRIVATE_KEY !== undefined ? [process.env.PRIVATE_KEY] : [],
    },
  },

  // Optional: Etherscan verification
  // Requires ETHERSCAN_API_KEY in .env
  etherscan: {
    apiKey: process.env.ETHERSCAN_API_KEY || "",
  },

  // Gas reporter (npm run test shows gas table)
  gasReporter: {
    enabled: process.env.REPORT_GAS === "true",
    currency: "USD",
  },
};
